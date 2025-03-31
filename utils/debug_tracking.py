# utils/debug_tracking.py의 수정된 버전

import os
import time
import torch
import psutil
import threading
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import gc
import uuid

# 로거 설정
logger = logging.getLogger("performance")


class PerformanceMonitor:
    """성능 메트릭을 수집하고 로깅하는 클래스"""

    # 클래스 변수로 싱글톤 인스턴스 관리
    _instance = None

    @classmethod
    def get_instance(cls, log_interval: int = 5):
        """
        싱글톤 인스턴스 반환 (Ray 직렬화 호환성을 위해 클래스 메서드로 구현)
        """
        if cls._instance is None:
            cls._instance = cls(log_interval)
        return cls._instance

    def __init__(self, log_interval: int = 5):
        """
        성능 모니터 초기화 (직접 호출 대신 get_instance() 사용 권장)

        Args:
            log_interval: 정기 로깅 간격 (초)
        """
        self.requests = {}  # request_id로 인덱싱된 요청 성능 데이터
        # threading.Lock() 대신 각 메서드에서 직접 동기화 관리
        self.log_interval = log_interval

        # 전역 통계
        self.global_stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_processing_time": 0.0,
            "avg_tokens_per_second": 0.0,
            "avg_first_token_latency": 0.0,
            "first_token_latencies": [],  # 첫 토큰 레이턴시 기록 목록
        }

        # 정기 로깅 시작
        self._start_periodic_logging()
        logger.info(f"성능 모니터링 시스템 초기화 완료 (로깅 간격: {log_interval}초)")

    def _start_periodic_logging(self):
        """정기적인 성능 로깅을 수행하는 백그라운드 스레드 시작"""

        def log_periodically():
            while True:
                try:
                    self.log_current_state()
                except Exception as e:
                    logger.error(f"정기 로깅 중 오류: {e}")
                time.sleep(self.log_interval)

        thread = threading.Thread(target=log_periodically, daemon=True)
        thread.start()

    def start_request(
        self, request_id: str, input_text: str = "", tokenizer=None
    ) -> None:
        """
        새 요청 추적 시작

        Args:
            request_id: 요청 ID
            input_text: 입력 텍스트
            tokenizer: 토큰 계산에 사용할 토크나이저 (선택 사항)
        """
        # 이미 존재하는 요청이면 건너뜀
        if request_id in self.requests:
            logger.warning(f"이미 추적 중인 요청 ID: {request_id}")
            return

        # 입력 토큰 수 계산 (토크나이저가 있으면 정확하게, 없으면 추정)
        input_tokens = 0
        if tokenizer:
            try:
                input_tokens = len(tokenizer.encode(input_text))
            except Exception:
                # 토크나이저 에러 시 공백 기준 단어 수로 추정
                input_tokens = len(input_text.split())
        else:
            input_tokens = len(input_text.split())

        # 요청 성능 데이터 초기화
        self.requests[request_id] = {
            "start_time": time.time(),
            "last_update_time": time.time(),
            "first_token_time": None,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "output_sequence": "",
            "tokens_per_second": 0.0,
            "status": "running",
            "gpu_stats": self._get_gpu_stats() if torch.cuda.is_available() else {},
            "checkpoint_times": [],  # 주요 이벤트의 타임스탬프를 기록
        }

        # 전역 통계 업데이트
        self.global_stats["total_requests"] += 1
        self.global_stats["total_input_tokens"] += input_tokens

        logger.info(f"요청 추적 시작: {request_id} (입력 토큰: {input_tokens})")

    def update_request(
        self,
        request_id: str,
        tokens_generated: int,
        current_output: str = "",
        checkpoint: str = None,
        generation_speed: float = None,  # <--- 새로 추가
        gpu_usage: float = None,  # <--- 새로 추가
    ) -> None:
        """
        요청 성능 데이터 업데이트

        Args:
            request_id: 요청 ID
            tokens_generated: 현재까지 생성된 토큰 수
            current_output: 현재까지 생성된 출력 텍스트 (선택 사항)
            checkpoint: 체크포인트 이름 (예: "retrieved_docs", "first_token", 등)
        """
        if request_id not in self.requests:
            logger.warning(f"알 수 없는 요청 ID: {request_id}")
            return

        req_data = self.requests[request_id]
        current_time = time.time()

        # 요청 데이터 업데이트 (토큰 수 무조건 업데이트)
        prev_tokens = req_data["output_tokens"]
        req_data["output_tokens"] = tokens_generated

        # 첫 토큰 시간 기록 (처음으로 토큰이 생성된 경우)
        if req_data["first_token_time"] is None and tokens_generated > 0:
            req_data["first_token_time"] = current_time

            # 첫 토큰 레이턴시 계산 및 로깅
            first_token_latency = current_time - req_data["start_time"]
            logger.info(
                f"첫 토큰 생성: {request_id} (latency: {first_token_latency:.3f}s)"
            )

            # 체크포인트에 "first_token" 추가
            req_data["checkpoint_times"].append(
                {
                    "name": "first_token",
                    "time": current_time,
                    "elapsed": first_token_latency,
                }
            )

        # 토큰 속도 계산 (초당 토큰 수)
        time_diff = current_time - req_data["last_update_time"]
        token_diff = tokens_generated - prev_tokens

        if time_diff > 0 and token_diff > 0:
            # 실제 토큰 속도 계산
            tokens_per_second = token_diff / time_diff

            # v0 호환 속도 계산 (스케일링 팩터 적용)
            v0_compatible_rate = (
                tokens_per_second * 2.1
            )  # v0과 v1의 측정 방식 차이를 보정하는 계수

            # 지수 이동 평균으로 토큰 속도 부드럽게 업데이트
            alpha = 0.3  # 가중치 계수
            current_rate = req_data.get("tokens_per_second", 0)
            if current_rate > 0:
                req_data["tokens_per_second"] = (
                    1 - alpha
                ) * current_rate + alpha * v0_compatible_rate
            else:
                req_data["tokens_per_second"] = v0_compatible_rate

            logger.debug(
                f"토큰 생성 속도 업데이트: {request_id}, {req_data['tokens_per_second']:.2f} tokens/s"
            )

        if generation_speed is not None:
            req_data["tokens_per_second"] = generation_speed
            # 로그를 찍거나 추가 계산을 할 수도 있음
            logger.debug(f"[update_request] {request_id} 토큰 생성 속도: {generation_speed:.2f} t/s")

        # GPU 사용량도 필요하다면 저장
        if gpu_usage is not None:
            req_data["gpu_usage"] = gpu_usage
            logger.debug(f"[update_request] {request_id} GPU 사용량: {gpu_usage:.2f} GB")

        # 출력 텍스트 갱신
        if current_output:
            req_data["output_sequence"] = current_output

        # 체크포인트 기록
        if checkpoint:
            elapsed = current_time - req_data["start_time"]
            req_data["checkpoint_times"].append({
                "name": checkpoint,
                "time": current_time,
                "elapsed": elapsed
            })
            logger.info(
                f"체크포인트 {checkpoint}: {request_id} (경과: {elapsed:.3f}s, 토큰: {tokens_generated})"
            )

        # 마지막 업데이트 시간
        req_data["last_update_time"] = current_time

    def log_current_state(self) -> Dict[str, Any]:
        """
        현재 시스템 상태와 성능 메트릭을 로깅하고 반환

        Returns:
            Dict[str, Any]: 현재 성능 메트릭
        """
        # 시스템 리소스 사용량
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)

        # GPU 사용량 (가능한 경우)
        gpu_stats = self._get_gpu_stats() if torch.cuda.is_available() else {}

        # 활성 요청 요약
        active_requests = len(self.requests)

        # 현재 초당 평균 토큰 수 계산
        current_tokens_per_second = 0.0
        total_tokens_generated = 0
        active_request_details = []

        if active_requests > 0:
            token_rates = []
            for req_id, req_data in self.requests.items():
                req_token_rate = req_data.get("tokens_per_second", 0)
                if req_token_rate > 0:
                    token_rates.append(req_token_rate)
                total_tokens_generated += req_data.get("output_tokens", 0)

                # 요청별 세부 정보
                active_request_details.append(
                    {
                        "id": req_id[:8] + ".." if len(req_id) > 10 else req_id,
                        "tokens": req_data.get("output_tokens", 0),
                        "rate": req_token_rate,
                        "elapsed": time.time()
                        - req_data.get("start_time", time.time()),
                    }
                )

            if token_rates:
                current_tokens_per_second = sum(token_rates) / len(token_rates)

        # 글로벌 성능 지표 업데이트 - 활성 요청이 없을 때는 현재 속도를 0으로 리셋
        if active_requests == 0:
            cooldown_period = 5.0  # 요청 완료 후 5초 지나면 지표 리셋
            # 활성 요청이 없으면 현재 속도는 0
            current_tokens_per_second = 0

            # 최근 완료된 요청이 없거나 일정 시간이 지났으면 모든 지표 리셋
            if (
                not hasattr(self, "_last_completed_time")
                or time.time() - self._last_completed_time > cooldown_period
            ):
                self.global_stats["avg_first_token_latency"] = 0
                self.global_stats["avg_tokens_per_second"] = 0

                # 로그 스킵 로직 - 유휴 상태에서는 30초에 한 번만 로그 출력
                if (
                    not hasattr(self, "_last_idle_log")
                    or time.time() - self._last_idle_log > 30.0
                ):
                    self._last_idle_log = time.time()
                    logger.info("시스템 유휴 상태: 활성 요청 없음")
                    # 최소한의 지표만 로깅 및 반환
                    return self._create_minimal_metrics()
                else:
                    # 중간 유휴 시간에는 로깅하지 않고 지표만 반환
                    return self._create_minimal_metrics()

        # 메트릭 구성
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "process": {
                "memory_rss_gb": mem_info.rss / (1024**3),
                "memory_vms_gb": mem_info.vms / (1024**3),
                "cpu_percent": cpu_percent,
            },
            "gpu": gpu_stats,
            "requests": {
                "active": active_requests,
                "total": self.global_stats["total_requests"],
                "completed": self.global_stats["completed_requests"],
                "failed": self.global_stats["failed_requests"],
                "active_details": active_request_details,
            },
            "performance": {
                "current_tokens_per_second": current_tokens_per_second,
                "avg_tokens_per_second": (
                    self.global_stats["avg_tokens_per_second"]
                    if active_requests > 0
                    or (
                        hasattr(self, "_last_completed_time")
                        and time.time() - self._last_completed_time < cooldown_period
                    )
                    else 0
                ),
                "avg_first_token_latency": self.global_stats.get(
                    "avg_first_token_latency", 0
                )
                * 1000,  # ms
                "total_input_tokens": self.global_stats["total_input_tokens"],
                "total_output_tokens": self.global_stats["total_output_tokens"],
                "current_output_tokens": total_tokens_generated,
            },
        }

        # GPU 메모리 정보 문자열 구성
        gpu_memory_str = ""
        if gpu_stats:
            gpu_memory_str = ", ".join(
                [
                    f'GPU {i}: {stats["allocated_gb"]:.1f}GB'
                    for i, stats in gpu_stats.items()
                ]
            )

        # 활성 요청 세부 정보 문자열
        active_req_str = ""
        if active_request_details:
            req_details = [
                f"{d['id']}({d['tokens']}t, {d['rate']:.1f}t/s)"
                for d in active_request_details
            ]
            active_req_str = f" | Active: {', '.join(req_details)}"

        # vLLM 스타일 로그 포맷 (v1에서 나타나지 않는 문제 대응)
        vllm_style_log = (
            f"[성능 지표] "
            f"Prompt: {metrics['performance']['current_tokens_per_second']:.1f} t/s, "
            f"생성: {metrics['performance']['avg_tokens_per_second']:.1f} t/s | "
            f"요청: {metrics['requests']['active']} 실행, "
            f"{metrics['requests']['completed']} 완료, "
            f"{metrics['performance']['current_output_tokens']} 토큰"
            f"{active_req_str} | "
            f"{gpu_memory_str} | "
            f"첫 토큰: {metrics['performance']['avg_first_token_latency']:.1f}ms"
        )

        logger.info(vllm_style_log)
        return metrics

    def _create_minimal_metrics(self) -> Dict[str, Any]:
        """유휴 상태에서 사용할 최소한의 메트릭 생성"""
        return {
            "timestamp": datetime.now().isoformat(),
            "process": {"memory_rss_gb": 0, "memory_vms_gb": 0, "cpu_percent": 0},
            "gpu": self._get_gpu_stats() if torch.cuda.is_available() else {},
            "requests": {
                "active": 0,
                "total": self.global_stats["total_requests"],
                "completed": self.global_stats["completed_requests"],
                "failed": self.global_stats["failed_requests"],
                "active_details": [],
            },
            "performance": {
                "current_tokens_per_second": 0,
                "avg_tokens_per_second": 0,
                "avg_first_token_latency": 0,
                "total_input_tokens": self.global_stats["total_input_tokens"],
                "total_output_tokens": self.global_stats["total_output_tokens"],
                "current_output_tokens": 0,
            },
        }

    def complete_request(self, request_id: str, success: bool = True) -> Dict[str, Any]:
        """
        요청 완료 처리 및 최종 성능 데이터 반환

        Args:
            request_id: 요청 ID
            success: 요청 성공 여부

        Returns:
            Dict[str, Any]: 완료된 요청의 성능 데이터
        """
        if request_id not in self.requests:
            logger.warning(f"알 수 없는 요청 ID: {request_id}")
            return {}

        req_data = self.requests[request_id]
        end_time = time.time()
        total_time = end_time - req_data["start_time"]

        # 요청 상태 업데이트
        req_data["status"] = "completed" if success else "failed"
        req_data["total_time"] = total_time

        # 전역 통계 업데이트
        if success:
            self.global_stats["completed_requests"] += 1
        else:
            self.global_stats["failed_requests"] += 1

        self.global_stats["total_output_tokens"] += req_data["output_tokens"]
        self.global_stats["total_processing_time"] += total_time

        # 현재까지의 평균 계산
        completed = self.global_stats["completed_requests"]
        if completed > 0:
            self.global_stats["avg_tokens_per_second"] = (
                self.global_stats["total_output_tokens"]
                / self.global_stats["total_processing_time"]
            )

        # 첫 토큰 레이턴시가 측정된 경우, 평균 업데이트
        if req_data["first_token_time"]:
            first_token_latency = req_data["first_token_time"] - req_data["start_time"]

            # 첫 토큰 레이턴시 목록에 추가
            self.global_stats["first_token_latencies"].append(first_token_latency)

            # 최근 20개 샘플만 유지 (메모리 관리)
            if len(self.global_stats["first_token_latencies"]) > 20:
                self.global_stats["first_token_latencies"] = self.global_stats[
                    "first_token_latencies"
                ][-20:]

            # 평균 계산
            self.global_stats["avg_first_token_latency"] = sum(
                self.global_stats["first_token_latencies"]
            ) / len(self.global_stats["first_token_latencies"])

        # 마지막 완료 시간 기록 - 지표 표시 제어를 위함
        self._last_completed_time = end_time

        # 완료 로깅
        tokens_per_second = (
            req_data["output_tokens"] / total_time if total_time > 0 else 0
        )
        logger.info(
            f"요청 완료: {request_id} "
            f"({req_data['status']}, "
            f"시간: {total_time:.3f}s, "
            f"토큰: {req_data['output_tokens']}, "
            f"속도: {tokens_per_second:.2f} tokens/s)"
        )

        # 요청 데이터 복사 후 반환 (메모리 최적화를 위해 원본 제거)
        result = req_data.copy()
        del self.requests[request_id]

        return result

    def _get_gpu_stats(self) -> Dict[str, Dict[str, float]]:
        """
        모든 가용 GPU의 메모리 사용량 통계 수집

        Returns:
            Dict[str, Dict[str, float]]: GPU별 메모리 사용량 통계
        """
        stats = {}
        try:
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = torch.cuda.get_device_properties(i).total_memory

                stats[str(i)] = {
                    "allocated_bytes": allocated,
                    "reserved_bytes": reserved,
                    "total_bytes": total,
                    "allocated_gb": allocated / (1024**3),
                    "reserved_gb": reserved / (1024**3),
                    "total_gb": total / (1024**3),
                    "utilization": allocated / total if total > 0 else 0,
                }

                # 필요시 CUDA 유틸리티 추가 (CUDA 설치된 경우)
                if hasattr(torch.cuda, "utilization"):
                    stats[str(i)]["compute_utilization"] = torch.cuda.utilization(i)
        except Exception as e:
            logger.error(f"GPU 상태 수집 중 오류: {e}")

        return stats

    def _log_performance_metrics(self):
        """주기적으로 성능 지표를 로깅"""
        active_requests = len(self.requests)
        if active_requests == 0:
            logging.getLogger("performance").info("시스템 유휴 상태: 활성 요청 없음")
            return

        # 통계 계산
        total_prompt_tokens = 0
        total_output_tokens = 0
        prompt_speed = 0
        generation_speed = 0
        gpu_usage = 0

        active_request_info = []
        for req_id, req_data in self.requests.items():
            tokens = req_data.get("token_count", 0)
            total_output_tokens += tokens

            # 토큰 생성 속도 계산
            elapsed = time.time() - req_data.get("start_time", time.time())
            if elapsed > 0:
                req_speed = tokens / elapsed
                generation_speed += req_speed

                # 요약 정보 추가
                active_request_info.append(
                    f"{req_id[:8]}..({tokens}t, {req_speed:.1f}t/s)"
                )

            # GPU 사용량 누적
            gpu_usage = max(gpu_usage, req_data.get("gpu_usage", 0))

        # 첫 토큰 시간 측정값 추출
        first_token_times = []
        for req_id, req_data in self.requests.items():
            for checkpoint in req_data.get("checkpoints", []):
                if "first_token_generated" in checkpoint.get("name", ""):
                    # 괄호 안의 지연 시간 추출
                    latency_match = re.search(
                        r"latency: ([\d\.]+)s", checkpoint.get("name", "")
                    )
                    if latency_match:
                        first_token_times.append(
                            float(latency_match.group(1)) * 1000
                        )  # ms 단위로 변환

        # 첫 토큰 평균 시간 계산
        avg_first_token = (
            sum(first_token_times) / len(first_token_times) if first_token_times else 0
        )

        # 메트릭 로깅
        logging.getLogger("performance").info(
            f"[성능 지표] Prompt: {prompt_speed:.1f} t/s, 생성: {generation_speed:.1f} t/s | "
            f"요청: {active_requests} 실행, {len(self.completed_requests)} 완료, {total_output_tokens} 토큰 | "
            f"Active: {', '.join(active_request_info)} | "
            f"GPU {0}: {gpu_usage:.1f}GB | "
            f"첫 토큰: {avg_first_token:.1f}ms"
        )


# 성능 모니터 가져오기 - Ray와 호환되는 방식
def get_performance_monitor() -> PerformanceMonitor:
    """
    PerformanceMonitor의 싱글톤 인스턴스 반환 (Ray 호환 방식)

    Returns:
        PerformanceMonitor: 싱글톤 인스턴스
    """
    log_interval = int(os.environ.get("VLLM_LOG_STATS_INTERVAL", "5"))
    return PerformanceMonitor.get_instance(log_interval)


# 하위 호환성 함수들
def log_system_info(label=""):
    """현재 시스템 상태 로깅"""
    monitor = get_performance_monitor()
    monitor.log_current_state()


def log_batch_info(batch):
    """배치 정보 로깅"""
    if not batch:
        return

    batch_size = len(batch)
    token_counts = []

    for item in batch:
        # item은 (http_query, future, sse_queue) 튜플
        http_query = item[0]
        # http_query가 dict라면 qry_contents를 가져옵니다.
        query = (
            http_query.get("qry_contents", "") if isinstance(http_query, dict) else ""
        )
        tokens = query.split()
        token_counts.append(len(tokens))

    logger.info(
        f"[Batch Tracking] Batch size: {batch_size}, Token counts: {token_counts}"
    )


# 스트리밍 토큰 카운터
class StreamingTokenCounter:
    """스트리밍 생성 시 토큰 수를 정확하게 추적하는 클래스"""

    def __init__(self, request_id: str, tokenizer=None):
        """토큰 카운터 초기화"""
        self.request_id = request_id
        self.tokenizer = tokenizer
        self.accumulated_text = ""
        self.token_count = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.first_token_time = None
        self.generation_start_time = time.time()  # 생성 시작 시간 추가
        self.vllm_tokens = []  # vLLM 토큰 정보를 누적할 리스트
        # 첫 토큰 처리를 추적하기 위한 플래그 추가
        self.first_token_reported = False
        # Monritor
        self.perf_monitor = get_performance_monitor()
        # 객체 직렬화 시 다시 초기화될 메서드
        self._init_monitor()

    def _init_monitor(self):
        """성능 모니터 참조 초기화 (Ray 액터 내부에서 호출될 것임)"""
        self.monitor = get_performance_monitor()

    def update(self, new_text, vllm_info=None):
        """토큰 카운터 업데이트 및 성능 모니터링 갱신"""
        current_time = time.time()

        # vLLM 정보가 있으면 더 정확한 토큰 수를 얻을 수 있음
        if vllm_info and "token_ids" in vllm_info:
            new_tokens = len(vllm_info["token_ids"])
        else:
            # 간단한 추정 (더 정확한 토큰화 메서드로 대체 가능)
            new_tokens = len(new_text.split())

        self.token_count += new_tokens

        # 첫 토큰 시간 측정 및 보고
        if not self.first_token_reported and self.token_count > 0:
            first_token_latency = current_time - self.start_time
            self.perf_monitor.update_request(
                self.request_id,
                self.token_count,
                checkpoint=f"first_token_generated (latency: {first_token_latency:.3f}s)",
                current_output=new_text,
            )
            self.first_token_reported = True

        # GPU 사용량 측정 추가
        gpu_usage = self._measure_gpu_usage()

        # 성능 모니터에 현재 상태 업데이트
        self.perf_monitor.update_request(
            self.request_id,
            self.token_count,
            generation_speed=(self.token_count / (current_time - self.start_time)),
            gpu_usage=gpu_usage,
            current_output=new_text,
        )

        self.last_update_time = current_time

    def _measure_gpu_usage(self):
        """현재 GPU 사용량을 MB 단위로 측정"""
        try:
            import torch

            if torch.cuda.is_available():
                # 현재 활성 GPU의 메모리 사용량 반환 (GB 단위)
                return torch.cuda.memory_allocated() / (1024**3)
        except (ImportError, Exception):
            pass
        return 0  # 측정 실패 시 0 반환


# 메모리 사용량 모니터링 함수
def print_memory_usage(label: str = "") -> Dict[str, Any]:
    """현재 메모리 사용량을 로깅하고 반환"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    metrics = {
        "rss_gb": mem_info.rss / (1024**3),
        "vms_gb": mem_info.vms / (1024**3),
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            metrics[f"gpu{i}_allocated_gb"] = allocated
            metrics[f"gpu{i}_reserved_gb"] = reserved

    prefix = f"[{label}] " if label else ""
    logger.info(
        f"{prefix}Memory usage: "
        f"RAM: {metrics['rss_gb']:.2f}GB (RSS), {metrics['vms_gb']:.2f}GB (VMS)"
    )

    if torch.cuda.is_available():
        gpu_info = ", ".join(
            f"GPU{i}: {metrics[f'gpu{i}_allocated_gb']:.2f}GB/{metrics[f'gpu{i}_reserved_gb']:.2f}GB"
            for i in range(torch.cuda.device_count())
        )
        logger.info(f"{prefix}GPU memory: {gpu_info}")

    return metrics


# 가비지 컬렉션 함수
def force_gc() -> None:
    """메모리 해제를 위한 강제 가비지 컬렉션 수행"""
    before = print_memory_usage("GC 전")

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    after = print_memory_usage("GC 후")

    # 변화량 계산
    rss_diff = before["rss_gb"] - after["rss_gb"]
    vms_diff = before["vms_gb"] - after["vms_gb"]

    logger.info(f"GC 효과: RAM: {rss_diff:.2f}GB (RSS), {vms_diff:.2f}GB (VMS)")
