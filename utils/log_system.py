# utils/log_system.py
from __future__ import annotations
import asyncio, logging, time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Deque, Any
import ray  # ★ MetricsCollectorActor 에서 사용
import re, math, json


# ─────────────────────────────────────────────────────────────────────────────
#  1)  Request‑level 데이터
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RequestMetrics:
    request_id: str
    query: str
    method_flags: str  # "rag:on,image:off,sql:on"
    start_ts: float = field(default_factory=time.time)
    first_token_ts: Optional[float] = None
    end_ts: Optional[float] = None
    prompt_tokens: int = 0
    generated_tokens: int = 0

    def mark_first_token(self):
        self.first_token_ts = self.first_token_ts or time.time()

    def mark_end(self):
        self.end_ts = self.end_ts or time.time()

    # ───────── helper properties ─────────
    @property
    def ttft(self):
        return (
            None if self.first_token_ts is None else self.first_token_ts - self.start_ts
        )

    @property
    def total_latency(self):
        return None if self.end_ts is None else self.end_ts - self.start_ts

    @property
    def gen_tps(self):
        if self.end_ts is None or self.first_token_ts is None:
            return None
        elap = self.end_ts - self.first_token_ts
        return self.generated_tokens / elap if elap > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
#  2)  MetricsManager  (배치 로그 OFF + idle 억제)
# ─────────────────────────────────────────────────────────────────────────────
class MetricsManager:
    _instance: Optional["MetricsManager"] = None

    @classmethod
    def get_instance(cls, *a, **kw) -> "MetricsManager":
        if cls._instance is None:
            cls._instance = cls(*a, **kw)
        return cls._instance

    def __init__(self, engine_async=None, stats_interval_sec: int = 5):
        if getattr(self, "_init", False):
            return
        self._init = True

        self._engine = engine_async
        self._stats_interval = stats_interval_sec

        # ───── 요청 데이터 ─────
        self._req: Dict[str, RequestMetrics] = {}                # 현재 진행중인 요청 데이터
        self._last50: Deque[RequestMetrics] = deque(maxlen=50)   # 마지막 50개 요청 데이터
        self._finished: list[RequestMetrics] = []                # ★ 모든 완료 요청

        # ───── “누적” 카운터 ─────
        self._total_requests: int = 0
        self._total_finished: int = 0
        self._total_generated_tokens: int = 0

        # ───── 로거 및 필터 설정 ─────
        self._logger = logging.getLogger("control_tower")
        self._logger.setLevel(logging.INFO)
        self._vllm_filter = _VllmIdleFilter(self)  # ← 필터 객체
        # >>> NEW : 모든 현존 logger 에 필터 부착
        self._attach_filter_to_all()
        # 이후 생길 logger 도 주기적으로 잡아주기 위해 set
        self._known_logger_ids = {id(l) for l in logging.root.manager.loggerDict.values()
                                  if isinstance(l, logging.Logger)}
        
        # ───── 기타 상태 ─────
        self._prev_stat: Optional[str] = None
        self._same_cnt = 0
        self._idle = False
        self._stats_task = None

    # ───────── public API ─────────
    def set_engine(self, engine_async):
        self._engine = engine_async

    def start(self, loop: asyncio.AbstractEventLoop = None):
        if self._stats_task is not None:
            return
        loop = loop or asyncio.get_event_loop()
        self._stats_task = loop.create_task(self._stats_ticker())

    # ---- per‑request ----
    def start_request(
        self, rid, q, *, prompt_tokens=0, rag=True, image=False, sql=False
    ):
        """요청이 들어올 때 호출"""
        flags = f"rag:{'on' if rag else 'off'},image:{'on' if image else 'off'},sql:{'on' if sql else 'off'}"
        self._req[rid] = RequestMetrics(rid, q, flags, prompt_tokens=prompt_tokens)
        self._total_requests += 1                                # ★ 누적 요청 수 증가

    def first_token(self, rid):
        self._req.get(rid) and self._req[rid].mark_first_token()

    def update_tokens(self, rid, n):
        self._req.get(rid) and setattr(self._req[rid], "generated_tokens", n)

    def finish_request(self, rid, answer_text=""):
        """응답이 끝났을 때 호출"""
        m = self._req.pop(rid, None)
        if not m:
            return
        m.mark_end()
        self._last50.append(m)
        self._finished.append(m)          # ★ 전체 목록에도 저장

        # ★ 누적 통계 업데이트
        self._total_finished += 1
        self._total_generated_tokens += m.generated_tokens

    # ---- dashboard 용 ----
    def dump_state(self) -> Dict[str, Any]:
        last50 = list(self._last50)
        finished_list = self._finished
        return {
            # ───────── 누적(global) 영역 ─────────
            "global": {
                "total_requests":          self._total_requests,
                "finished_requests":       self._total_finished,
                "unfinished_requests":     len(self._req),
                "total_generated_tokens":  self._total_generated_tokens,
                # ── 최근 50건 평균 ──
                "avg_gen_tps_last50":   sum((m.gen_tps or 0) for m in last50) / (len(last50) or 1),
                "avg_ttft_last50":      sum((m.ttft or 0) for m in last50) / (len(last50) or 1),
                "avg_total_time_last50": sum((m.total_latency or 0) for m in last50) / (len(last50) or 1),
            },
            # ───────── 상세 목록 ─────────
            "active_requests":   {rid: m.__dict__ for rid, m in self._req.items()},
            "recent_finished":   [m.__dict__ for m in last50],     # 최근 50건 종료
            "all_finished": [m.__dict__ for m in finished_list],   # 전체 종료
        }

    # ------------------------------------------------------------
    # NEW : 필터를 (중복 없이) 모든 logger 에 붙이는 헬퍼
    # ------------------------------------------------------------
    def _attach_filter_to_all(self):
        root_logger = logging.getLogger()
        if self._vllm_filter not in root_logger.filters:
            root_logger.addFilter(self._vllm_filter)

        for lg in logging.root.manager.loggerDict.values():
            if isinstance(lg, logging.Logger) and self._vllm_filter not in lg.filters:
                lg.addFilter(self._vllm_filter)

    # ───────── internal : stats ticker ─────────
    async def _stats_ticker(self):
        idle_ticks = 0
        while True:
            await asyncio.sleep(self._stats_interval)

            # --- 새 logger 가 생겼는지 확인해 필터 재부착 -------------
            cur_ids = {id(l) for l in logging.root.manager.loggerDict.values()
                       if isinstance(l, logging.Logger)}
            if cur_ids - self._known_logger_ids:
                self._attach_filter_to_all()
                self._known_logger_ids = cur_ids
            # ---------------------------------------------------------

            # ---------- idle 판정 ----------
            if len(self._req) == 0:
                idle_ticks += 1
            else:
                idle_ticks = 0

            self._idle = idle_ticks >= 3          # ≥15 s 무요청 → idle

            if self._idle:
                continue                          # idle 이면 do_log_stats 호출 생략

            stat = await _safe_get_stats(self._engine)
            if stat is None:
                continue
            if isinstance(stat, (dict, list)):
                stat = json.dumps(stat, ensure_ascii=False)
            self._logger.info(stat.strip())

    # idle 상태가 바뀔 때마다 필터에 알려줌
    @property
    def _idle(self):
        return getattr(self, "__idle", False)

    @_idle.setter
    def _idle(self, val: bool):
        self.__idle = val
        self._vllm_filter.idle = val


# ───────── helper : 안전하게 통계 문자열 얻기 ─────────
async def _safe_get_stats(engine):
    try:
        if hasattr(engine, "get_log_stats"):
            return await engine.get_log_stats()
        await engine.do_log_stats()
        return None
    except Exception:
        return None


# ───────── logging.Filter 구현 ─────────
class _VllmIdleFilter(logging.Filter):
    _pat = re.compile(r"Engine \d{3}:") # 필요하면 r"Engine \d+:" 로 확장

    def __init__(self, mgr: "MetricsManager"):
        super().__init__()
        self.mgr = mgr

    def filter(self, record: logging.LogRecord) -> bool:
        # idle 이면 vLLM 통계 차단
        if self.mgr._idle and self._pat.match(record.getMessage()):
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  3)  MetricsCollectorActor  (전역 집계용 – Ray 액터)
# ─────────────────────────────────────────────────────────────────────────────
@ray.remote
class MetricsCollectorActor:
    """
    여러 InferenceActor 가 보내는 지표를 모아 전역 집계를 수행.
    *필수 기능만 남겨 최소 구현*.
    """

    def __init__(self, log_interval_sec: int = 60):
        self._log_interval = log_interval_sec
        self._last200: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._logger = logging.getLogger("global_metrics")
        self._logger.setLevel(logging.INFO)
        asyncio.get_event_loop().create_task(self._ticker())

    # InferenceActor 가 호출
    def register_actor_request(self, actor_id: str, req_metrics: Dict[str, Any]):
        self._last200.append(req_metrics)

    def register_actor_stats(self, actor_id: str, stats: Dict[str, Any]):
        # 필요하다면 액터별 최신 상태를 저장할 수 있음
        pass

    # 집계 로그
    async def _ticker(self):
        while True:
            await asyncio.sleep(self._log_interval)
            self._logger.info("[GLOBAL] processed=%d (window 200)", len(self._last200))

    # 대시보드용
    def dump_global(self):
        return list(self._last200)
