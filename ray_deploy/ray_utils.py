# ray_deploy/ray_utils.py
import ray  # Ray library
from ray import serve
import json
import asyncio  # async I/O process module
from concurrent.futures import ProcessPoolExecutor  # 스레드 컨트롤
import uuid
import time
from typing import Dict, Optional, List, Any, Tuple, Union
import threading  # To find out the usage of thread
import datetime
import logging

from core import (
    query_sort,
    specific_question,
    execute_rag,
    generate_answer,
    generate_answer_stream,
    image_query,
    test_prompt_with_image,
    test_prompt_streaming,
)
from utils import (
    load_model,
    load_data,
    process_format_to_response,
    process_to_format,
    error_format,
)
from utils.summarizer import summarize_conversation
from utils.log_system import MetricsManager
# Langchain Memory system
from ray_deploy.langchain import CustomConversationBufferMemory, serialize_message

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RAY_UTILS')

# Configuration
import yaml
from box import Box
with open("./config.yaml", "r") as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)

@ray.remote  # From Decorator, Each Actor is allocated 1 GPU
class InferenceActor:
    def __init__(self, config, metrics_collector=None):
        """
        InferenceActor 초기화: 모델, 토크나이저, 데이터 로드 및 배치 처리 설정
        """
        self.config = config
        self.metrics_collector = metrics_collector
        
        # 액터 내부에서 모델 및 토크나이저를 새로 로드 (GPU에 한 번만 로드)
        self.model, self.tokenizer, self.embed_model, self.embed_tokenizer = load_model(
            config
        )
        # 데이터는 캐시 파일을 통해 로드
        self.data = load_data(config.data_path)
        
        # 비동기 큐와 배치 처리 설정
        self.request_queue = asyncio.Queue()
        self.max_batch_size = config.ray.max_batch_size  # 최대 배치 수
        self.batch_wait_timeout = config.ray.batch_wait_timeout  # 배치당 처리 시간 - 이제 사용하지 아니함(연속 배치이기에 배치당 처리 시간이 무의미)

        # Actor 내부에서 ProcessPoolExecutor 생성 (직렬화 문제 회피)
        max_workers = int(min(config.ray.num_cpus * 0.8, (26*config.ray.actor_count)-4))
        self.process_pool = ProcessPoolExecutor(max_workers)

        # --- SSE Queue Manager --- 
        self.queue_manager = ray.get_actor("SSEQueueManager")

        # --- 기타 설정 ---
        # 활성 작업 추적을 위한 변수 추가
        self.batch_counter = 0  # New counter to track batches
        self.active_tasks = set()
        self.max_concurrent_tasks = config.ray.max_batch_size
        # Track active generations that can be stopped
        self.active_generations = set()
        
        # 메모리
        self.memory_map = {}
        
        # 연속 배치 처리기 시작 (Continuous batch)
        asyncio.create_task(self.continuous_batch_processor())
        
        # MetricsManager 초기화 (더 안전하게)
        try:
            # 각 액터마다 자체 인스턴스 생성
            self.metrics = MetricsManager()
            # 엔진 설정
            self.metrics.set_engine(self.model)
            
            # 현재 이벤트 루프 가져와서 메트릭스 매니저 시작
            loop = asyncio.get_event_loop()
            self.metrics.start(loop)
            
            # 메트릭스 동기화 태스크 시작 (metrics_collector가 있는 경우)
            if self.metrics_collector is not None:
                asyncio.create_task(self._sync_metrics_with_collector())
        except Exception as e:
            logging.error(f"메트릭스 초기화 실패: {e}")
            # 오류 발생 시 더미 메트릭스 매니저 사용
            class DummyMetrics:
                def start_request(self, *args, **kwargs): pass
                def first_token(self, *args, **kwargs): pass
                def update_tokens(self, *args, **kwargs): pass
                def finish_request(self, *args, **kwargs): pass
            self.metrics = DummyMetrics()
            
    # -------------------------------------------------------------------------
    # GET MEMORY FOR SESSION - 세션별 메모리 관리
    # -------------------------------------------------------------------------
    def get_memory_for_session(self, request_id: str) -> CustomConversationBufferMemory:
        """
        세션별 Memory를 안전하게 가져오는 헬퍼 메서드.
        만약 memory_map에 request_id가 없으면 새로 생성해 저장 후 반환.
        
        Args:
            request_id: 세션(대화) ID
            
        Returns:
            CustomConversationBufferMemory: 대화 기록을 저장하는 메모리 객체
        """
        if request_id not in self.memory_map:
            logger.info(f"Creating new ConversationBufferMemory for session={request_id}")
            self.memory_map[request_id] = CustomConversationBufferMemory(return_messages=True)
        return self.memory_map[request_id]

    # -------------------------------------------------------------------------
    # CONTINUOUOS BATCH PROCESSOR - 연속 배치 처리
    # -------------------------------------------------------------------------
    async def continuous_batch_processor(self):
        """
        연속적인 배치 처리 - 최대 max_batch_size개의 요청이 항상 동시에 처리되도록 함
        """
        while True:
            # 사용 가능한 슬롯 확인
            available_slots = self.max_concurrent_tasks - len(self.active_tasks)
            
            if available_slots > 0:
                try:
                    # 새 요청 받기
                    request_obj, fut = await asyncio.wait_for(self.request_queue.get(), timeout=0.01)

                    # 비동기 처리 작업 생성
                    task = asyncio.create_task(
                        self._process_single_query(request_obj, fut)
                    )
                    
                    # 작업 완료 시 활성 목록에서 제거하는 콜백 설정
                    task.add_done_callback(
                        lambda t, task_ref=task: self.active_tasks.discard(task_ref)
                    )
                    
                    # 활성 작업 목록에 추가
                    self.active_tasks.add(task)
                    
                    logger.info(f"[Continuous Batching] +1 request => active tasks now {len(self.active_tasks)}/{self.max_concurrent_tasks}")
                
                except asyncio.TimeoutError:
                    # 새 요청이 없으면 잠시 대기
                    await asyncio.sleep(0.01)
            else:
                # 모든 슬롯이 사용 중이면 작업 완료될 때까지 대기
                if self.active_tasks:
                    # 작업 중 하나라도 완료될 때까지 대기
                    done, pending = await asyncio.wait(
                        self.active_tasks, 
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # 완료된 작업 확인
                    for task in done:
                        try:
                            await task
                        except Exception as e:
                            logger.error(f"Task failed: {e}")
                    
                    logger.info(f"[Continuous Batching] Tasks completed: {len(done)} => active tasks now {len(self.active_tasks)}/{self.max_concurrent_tasks}")
                else:
                    await asyncio.sleep(0.01)

    # -------------------------------------------------------------------------
    # 유틸리티 함수들: 코드 중복 제거 및 재사용성 향상
    # -------------------------------------------------------------------------
    def _get_request_info(self, http_query_or_stream_dict: dict) -> Tuple[dict, str, bool]:
        """
        요청 정보를 추출하는 유틸리티 함수
        
        Args:
            http_query_or_stream_dict: 원본 요청 데이터
            
        Returns:
            Tuple[dict, str, bool]: (http_query, request_id, is_streaming) 형태의 튜플
        """
        # 스트리밍 여부 및 request_id 확인
        if isinstance(http_query_or_stream_dict, dict) and "request_id" in http_query_or_stream_dict:
            request_id = http_query_or_stream_dict["request_id"]
            http_query = http_query_or_stream_dict["http_query"]
            is_streaming = True
            logger.info(f"[STREAM] request_id={request_id}")
        else:
            request_id = None
            http_query = http_query_or_stream_dict
            is_streaming = False
            logger.info("[NORMAL] Non-streaming request")
            
        return http_query, request_id, is_streaming
    
    async def _load_conversation_history(self, page_id: str) -> str:
        """
        대화 기록 로드하는 유틸리티 함수
        
        Args:
            page_id: 세션(대화) ID
            
        Returns:
            str: 포맷팅된 대화 이력 텍스트
        """
        memory = self.get_memory_for_session(page_id)
        
        try:
            past_context = memory.load_memory_variables({}).get("history", [])
            # history가 리스트 형식인 경우 (각 메시지가 별도 항목으로 저장되어 있다면)
            if isinstance(past_context, list):
                recent_messages = [msg if isinstance(msg, str) else msg.content for msg in past_context[-5:]]
                return "\n\n".join(recent_messages)
            else:
                # 문자열인 경우, 메시지 구분자를 "\n\n"으로 가정하여 분리
                messages = str(past_context).split("\n\n")
                recent_messages = messages[-5:]
                return "\n\n".join(recent_messages)
        except Exception as e:
            logger.error(f"대화 기록 로드 중 오류: {e}")
            return ""
        
    async def _process_image_if_exists(self, http_query: dict) -> dict:
        """
        이미지 데이터가 있는 경우 이미지 처리 수행
        
        Args:
            http_query: 요청 데이터
            
        Returns:
            dict: 이미지 설명 정보
        """
        image_data = http_query.get("image_data")
        if image_data is not None:
            logger.info("이미지 데이터 감지, 이미지 처리 시작")
            return await image_query(http_query, self.model, self.config)
        return {"is_structured": False, "description": "이미지는 입력되지 않았습니다."}
    
    async def _store_conversation(self, memory: CustomConversationBufferMemory, 
                                user_input: str, output: str, 
                                chunk_ids: List[str], http_query: dict) -> None:
        """
        대화 내용을 메모리에 저장하는 유틸리티 함수
        
        Args:
            memory: 대화 메모리 객체
            user_input: 사용자 입력
            output: AI 응답
            chunk_ids: 참조된 청크 ID 목록
            http_query: 원본 요청 데이터
        """
        try:
            memory.save_context(
                {
                    "qry_contents": user_input,
                    "qry_id": http_query.get("qry_id", ""),
                    "user_id": http_query.get("user_id", ""),
                    "auth_class": http_query.get("auth_class", ""),
                    "qry_time": http_query.get("qry_time", "")
                },
                {
                    "output": output,
                    "chunk_ids": chunk_ids
                }
            )
            logger.debug(f"대화 저장 완료: 청크 ID {chunk_ids}")
        except Exception as e:
            logger.error(f"대화 저장 중 오류: {e}")
    
    def _extract_chunk_ids(self, retrieval: dict) -> List[str]:
        """
        검색 결과에서 청크 ID 추출하는 유틸리티 함수
        
        Args:
            retrieval: 검색 결과 데이터
            
        Returns:
            List[str]: 청크 ID 목록
        """
        chunk_ids = []
        for doc in retrieval.get("rsp_data", []):
            if "chunk_id" in doc:
                chunk_ids.append(doc["chunk_id"])
        return chunk_ids
    
    async def _handle_error(self, e: Exception, request_id: str, future: asyncio.Future) -> None:
        """
        오류 처리를 위한 유틸리티 함수
        
        Args:
            e: 발생한 예외
            request_id: 요청 ID
            future: 결과를 설정할 Future 객체
        """
        err_msg = f"처리 중 오류 발생: {str(e)}"
        logger.error(err_msg)
        
        # 스트리밍인 경우 에러 토큰 전송
        if request_id:
            try:
                error_token = json.dumps({"type": "error", "message": err_msg}, ensure_ascii=False)
                await self.queue_manager.put_token.remote(request_id, error_token)
                await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")
                await self.queue_manager.delete_queue.remote(request_id)
            except Exception as e2:
                logger.error(f"SSE 에러 전송 중 추가 오류: {str(e2)}")
                
        # Future에 에러 결과 설정
        future.set_result(error_format(err_msg, 500))
    
    # -------------------------------------------------------------------------
    # 메인 처리 함수: RAG 활용 쿼리 처리
    # -------------------------------------------------------------------------
    async def _process_rag_query(self, http_query: dict, user_input: str, 
                               image_description: dict, request_id: str, 
                               is_streaming: bool, future: asyncio.Future) -> None:
        """
        RAG를 사용하는 쿼리 처리 로직
        
        Args:
            http_query: 요청 데이터
            user_input: 사용자 입력
            image_description: 이미지 설명 데이터
            request_id: 요청 ID
            is_streaming: 스트리밍 여부
            future: 결과를 설정할 Future 객체
        """
        try:
            # 데이터 다시 로드 (항상 최신 데이터 사용)
            self.data = load_data(self.config.data_path)
            
            # 질문 분류 및 RAG 파라미터 준비
            params = {
                "user_input": f"사용자 질문: {user_input} [이미지 설명: {image_description.get('description')}]",
                "model": self.model,
                "tokenizer": self.tokenizer,
                "embed_model": self.embed_model,
                "embed_tokenizer": self.embed_tokenizer,
                "data": self.data,
                "config": self.config,
                "request_id": request_id,
            }
            
            # 질문 구체화
            QU, KE, TA, TI = await query_sort(params)
            logger.info(f"질문 구체화 결과: QU={QU}, KE={KE}, TA={TA}, TI={TI}")
            
            # 테이블 데이터가 필요한 경우 (SQL 기반 RAG)
            if TA == "yes":
                await self._process_table_query(QU, KE, TA, TI, user_input, request_id, is_streaming, future, http_query)
            else:
                # 일반 RAG 처리
                await self._process_standard_rag(QU, KE, TA, TI, user_input, request_id, is_streaming, future, http_query)
                
        except Exception as e:
            await self._handle_error(e, request_id, future)
    
    # -------------------------------------------------------------------------
    # SQL 테이블 기반 처리 함수
    # -------------------------------------------------------------------------
    async def _process_table_query(self, QU: str, KE: str, TA: str, TI: str,
                                 user_input: str, request_id: str,
                                 is_streaming: bool, future: asyncio.Future, 
                                 http_query: dict) -> None:
        """
        테이블 데이터가 필요한 쿼리 처리 (SQL 기반)
        
        Args:
            QU, KE, TA, TI: 질문 분류 결과
            user_input: 사용자 입력
            request_id: 요청 ID
            is_streaming: 스트리밍 여부
            future: 결과를 설정할 Future 객체
            http_query: 원본 요청 데이터
        """
        try:
            # SQL 실행 및 데이터 추출
            docs, docs_list = await execute_rag(
                QU, KE, TA, TI,
                model=self.model,
                tokenizer=self.tokenizer,
                embed_model=self.embed_model,
                embed_tokenizer=self.embed_tokenizer,
                data=self.data,
                config=self.config,
            )
            
            # 포맷 변환
            try:
                retrieval, chart = process_to_format(docs_list, type="SQL")
            except Exception as e:
                logger.error(f"SQL 결과 포맷 변환 실패: {str(e)}")
                retrieval, chart = [], None
                
            # 스트리밍 또는 일반 응답 처리
            if is_streaming:
                await self._stream_partial_answer(QU, docs, retrieval, chart, request_id, future, user_input, http_query)
            else:
                # 일반 최종 결과 생성
                output = await generate_answer(QU, docs, model=self.model, tokenizer=self.tokenizer, config=self.config)
                answer = process_to_format([output, chart], type="Answer")
                final_data = [retrieval, answer]
                outputs = process_format_to_response(final_data, qry_id=None, continue_="C")
                
                # 청크 ID 추출 및 대화 저장
                chunk_ids = self._extract_chunk_ids(retrieval)
                memory = self.get_memory_for_session(http_query.get("page_id", request_id))
                await self._store_conversation(memory, user_input, output, chunk_ids, http_query)
                
                # 최종 결과 반환
                future.set_result(outputs)
                
        except Exception as e:
            await self._handle_error(e, request_id, future)
            
    # -------------------------------------------------------------------------
    # 표준 RAG 처리 함수
    # -------------------------------------------------------------------------
    async def _process_standard_rag(self, QU: str, KE: str, TA: str, TI: str,
                                  user_input: str, request_id: str,
                                  is_streaming: bool, future: asyncio.Future, 
                                  http_query: dict) -> None:
        """
        일반 RAG 처리 로직
        
        Args:
            QU, KE, TA, TI: 질문 분류 결과
            user_input: 사용자 입력
            request_id: 요청 ID
            is_streaming: 스트리밍 여부
            future: 결과를 설정할 Future 객체
            http_query: 원본 요청 데이터
        """
        try:
            # 대화 이력 기반 질문 구체화
            QU, KE, TA, TI = await specific_question({
                "user_input": user_input,
                "model": self.model,
                "tokenizer": self.tokenizer,
                "embed_model": self.embed_model,
                "embed_tokenizer": self.embed_tokenizer,
                "data": self.data,
                "config": self.config,
                "request_id": request_id,
            })
            
            # RAG 실행
            docs, docs_list = await execute_rag(
                QU, KE, TA, TI,
                model=self.model,
                tokenizer=self.tokenizer,
                embed_model=self.embed_model,
                embed_tokenizer=self.embed_tokenizer,
                data=self.data,
                config=self.config,
            )
            
            # 검색 결과 포맷 변환
            retrieval = process_to_format(docs_list, type="Retrieval")
            
            # 스트리밍 또는 일반 응답 처리
            if is_streaming:
                await self._stream_partial_answer(QU, docs, retrieval, None, request_id, future, user_input, http_query)
            else:
                # 최종 답변 생성
                output = await generate_answer(QU, docs, model=self.model, tokenizer=self.tokenizer, config=self.config)
                answer = process_to_format([output], type="Answer")
                final_data = [retrieval, answer]
                outputs = process_format_to_response(final_data, qry_id=None, continue_="C")
                
                # 청크 ID 추출 및 대화 저장
                chunk_ids = self._extract_chunk_ids(retrieval)
                memory = self.get_memory_for_session(http_query.get("page_id", request_id))
                await self._store_conversation(memory, user_input, output, chunk_ids, http_query)
                
                # 최종 결과 반환
                future.set_result(outputs)
                
        except Exception as e:
            await self._handle_error(e, request_id, future)
            
    # -------------------------------------------------------------------------
    # RAG 없이 직접 응답 생성 함수
    # -------------------------------------------------------------------------
    async def _process_direct_query(self, user_input: str, request_id: str, 
                                  future: asyncio.Future, http_query: dict) -> None:
        """
        RAG를 사용하지 않고 직접 응답을 생성하는 로직
        
        Args:
            user_input: 사용자 입력
            request_id: 요청 ID
            future: 결과를 설정할 Future 객체
            http_query: 원본 요청 데이터
        """
        try:
            # 이 함수는 RAG를 사용하지 않고 직접 응답을 생성합니다
            docs = None
            retrieval = None
            chart = None
            
            # 스트리밍 방식으로 응답 생성
            await self._stream_partial_answer(user_input, docs, retrieval, chart, request_id, future, user_input, http_query)
        except Exception as e:
            await self._handle_error(e, request_id, future)
    
    # -------------------------------------------------------------------------
    # PROCESS SINGLE QUERY (분기 처리 단순화)
    # -------------------------------------------------------------------------
    async def _process_single_query(self, http_query_or_stream_dict: dict, future: asyncio.Future) -> None:
        """
        단일 쿼리 처리의 메인 함수 (개선된 버전)
        
        Args:
            http_query_or_stream_dict: 원본 요청 데이터
            future: 결과를 설정할 Future 객체
            sse_queue: 스트리밍을 위한 SSE 큐
        """
        # 요청 정보 추출
        http_query, request_id, is_streaming = self._get_request_info(http_query_or_stream_dict)
        
        # 요청 ID가 없으면 생성 (성능 모니터링용)
        if not request_id:
            request_id = str(uuid.uuid4())
            
        # 사용자 입력 추출
        user_input = http_query.get("qry_contents", "")
        logger.info(f"사용자 입력: {user_input}")
        
        # RAG 사용 여부 확인
        use_rag = http_query.get("use_rag", True)
        
        # 이미지 데이터 확인
        image_data = http_query.get("image_data")
        
        try:
            # request_id 결정 후 - metrics 에 요청 시작 정보 전달
            self.metrics.start_request(
                request_id,
                user_input,
                rag=use_rag,
                image=bool(image_data),
                sql=False  # 나중에 테이블 쿼리면 True 로 다시 호출해도 됨
            )
            
            if use_rag is False:
                logger.info("RAG 비활성화 모드로 실행")
                await self._process_direct_query(user_input, request_id, future, http_query)
            else:
                # 이미지 처리 (있는 경우)
                image_description = await self._process_image_if_exists(http_query)
                
                # RAG 활용 응답 생성
                await self._process_rag_query(http_query, user_input, image_description, request_id, is_streaming, future)
        except Exception as e:
            err_msg = f"처리 중 오류 발생: {str(e)}"
            logger.error(err_msg)
            
            # 빈 응답으로 요청 종료 처리
            self.metrics.finish_request(request_id, answer_text="")
            
            # 스트리밍인 경우 에러 토큰 전송
            if is_streaming:
                try:
                    error_token = json.dumps({"type": "error", "message": err_msg}, ensure_ascii=False)
                    await self.queue_manager.put_token.remote(request_id, error_token)
                    await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")
                    await self.queue_manager.delete_queue.remote(request_id)
                except Exception as e2:
                    logger.error(f"SSE 에러 전송 중 추가 오류: {str(e2)}")
                    
            # Future에 에러 결과 설정
            future.set_result(error_format(err_msg, 500))
            
        finally:
            # 요청 중지 시에 active_generation 정리
            self.active_generations.discard(request_id)

            # 스트리밍 요청인 경우 정리 작업
            if is_streaming:
                try:
                    await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")
                except Exception as ex:
                    logger.error(f"STREAM_DONE 토큰 전송 중 오류: {str(ex)}")
                await self.queue_manager.delete_queue.remote(request_id)

    # -------------------------------------------------------------------------
    # STREAMING PARTIAL ANSWER - 스트리밍 응답 처리
    # -------------------------------------------------------------------------
    async def _stream_partial_answer(self, QU: str, docs, retrieval, chart, 
                                request_id: str, future: asyncio.Future, 
                                user_input: str, http_query: dict) -> None:
        """
        스트리밍 방식의 부분 응답 생성 및 전송
        
        Args:
            QU: 구체화된 질문
            docs: RAG 검색 문서 내용
            retrieval: 검색 결과 포맷
            chart: 차트 데이터 (있는 경우)
            request_id: 요청 ID
            future: 결과를 설정할 Future 객체
            user_input: 원본 사용자 입력
            http_query: 원본 요청 데이터
        """
        logger.info(f"스트리밍 응답 시작: request_id={request_id}")
        
        # Add to active generations set
        self.active_generations.add(request_id)
        
        # 성능 측정 초기화 - 시작 시간 기록
        stream_start_time = time.time()
        token_count = 0
        first_token_received = False

        try:
            # 1. 참조 데이터 전송
            if retrieval:
                reference_json = json.dumps({
                    "type": "reference",
                    "status_code": 200,
                    "result": "OK",
                    "detail": "Reference data",
                    "evt_time": datetime.datetime.now().isoformat(),
                    "data_list": [retrieval]
                }, ensure_ascii=False)
                await self.queue_manager.put_token.remote(request_id, reference_json)
                logger.info(f"참조 데이터 전송 완료: request_id={request_id}")
            
            # 2. 메모리 가져오기
            memory = self.get_memory_for_session(request_id)
            
            # 3. 대화 이력 불러오기
            past_context = await self._load_conversation_history(request_id)
            
            # 4. 최종 프롬프트 구성
            final_query = f"{past_context}\n\n[사용자 질문]\n{QU}"
            
            # 토큰 수 계산
            retrieval_str = str(retrieval) if retrieval else ""
            past_tokens = self.tokenizer.tokenize(str(past_context))
            query_tokens = self.tokenizer.tokenize(str(QU))
            retrieval_tokens = self.tokenizer.tokenize(retrieval_str)
            total_tokens = len(self.tokenizer.tokenize(str(final_query))) + len(retrieval_tokens)
            logger.info(f"토큰 수: 이전 대화={len(past_tokens)}, 검색 자료={len(retrieval_tokens)}, 질문={len(query_tokens)}, 총={total_tokens}")
                
            # 5. 스트리밍 응답 생성 및 전송
            partial_accumulator = ""
            
            # HTTP 쿼리에 request_id 추가하여 전달 - vLLM 토큰 추적용
            http_query.update({"page_id": request_id})
            
            async for partial_text in generate_answer_stream(
                final_query, docs, self.model, self.tokenizer, self.config, http_query
            ):
                new_text = partial_text[len(partial_accumulator):]
                partial_accumulator = partial_text
                
                if new_text == "":
                    continue
                
                # 토큰 수는 이제 StreamingTokenCounter에서 직접 추적되므로 여기서는 간단히 갱신
                current_tokens = len(partial_accumulator.split())
                token_count = current_tokens
                
                # 첫 토큰 도착 확인
                current_time = time.time()
                if not first_token_received and token_count > 0:
                    first_token_latency = current_time - stream_start_time
                    first_token_received = True
                    logger.info(f"첫 토큰 생성: {request_id} (latency: {first_token_latency:.3f}s)")
                    # 첫 토큰 생성 시간 기록
                    self.metrics.first_token(request_id)
                
                # 토큰 업데이트
                self.metrics.update_tokens(request_id, token_count)
                
                # 응답 토큰 JSON 형식으로 포장하여 전송
                answer_json = json.dumps({
                    "type": "answer",
                    "answer": new_text
                }, ensure_ascii=False)
                await self.queue_manager.put_token.remote(request_id, answer_json)
            
            # 생성 완료 시간 계산
            generation_time = time.time() - stream_start_time
            tokens_per_second = token_count / generation_time if generation_time > 0 else 0
            
            # 최종 축적된 텍스트
            final_text = partial_accumulator
            
            # 대화 저장
            chunk_ids = self._extract_chunk_ids(retrieval) if retrieval else []
            await self._store_conversation(memory, user_input, final_text, chunk_ids, http_query)
            
            # 최종 응답 구조 생성
            if chart is not None:
                ans = process_to_format([final_text, chart], type="Answer")
            else:
                ans = process_to_format([final_text], type="Answer")
                
            # 결과 설정 및 스트리밍 종료 신호 전송
            final_res = process_format_to_response([retrieval, ans] if retrieval else [ans], qry_id=http_query.get("qry_id", ""), continue_="E")
            future.set_result(final_res)
            
            # 요청 완료 처리 - 성공
            self.metrics.finish_request(request_id, final_text)
            
            logger.info(f"스트리밍 응답 완료: request_id={request_id}")
            
            # At the end, when generation completes naturally
            self.active_generations.discard(request_id)
            
            # ★ 전역 수집기에 보고1
            self.metrics.finish_request(request_id, final_text)

            # ★ 전역 수집기에 보고2
            if self.metrics_collector is not None:
                try:
                    await self.metrics_collector.register_actor_request.remote(
                        self.actor_id,            # ← __init__ 에서 만든 고유 ID
                        self.metrics.dump_state()["recent_finished"][-1]
                    )
                except Exception as e:
                    logger.warning(f"metrics report failed: {e}")
            
        except Exception as e:
            err_msg = f"스트리밍 응답 중 오류: {str(e)}"
            
            # generation completes whatever it stopped
            self.active_generations.discard(request_id)
            logger.error(err_msg)
            future.set_result(error_format(err_msg, 500))
            await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")

    # -------------------------------------------------------------------------
    # API 인터페이스 메서드들
    # -------------------------------------------------------------------------
    async def process_query(self, http_query: dict):
        """
        일반 쿼리 처리 메서드 - 최종 결과를 한 번에 반환
        
        Args:
            http_query: 요청 데이터
            
        Returns:
            dict: 처리 결과
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        # 스트리밍 아님 - sse_queue는 None
        await self.request_queue.put((http_query, future, None))
        return await future
    
    async def process_query_stream(self, http_query: dict) -> str:
        """
        스트리밍 쿼리 처리 메서드 - SSE 큐를 생성하고 request_id 반환
        
        Args:
            http_query: 요청 데이터
            
        Returns:
            str: 생성된 채팅/요청 ID
        """
        # page_id를 채팅방 id로 사용 (없으면 생성)
        chat_id = http_query.get("page_id")
        if not chat_id:
            chat_id = str(uuid.uuid4())
        http_query["page_id"] = chat_id  # 강제 할당
        
        # SSE 큐 생성
        await self.queue_manager.create_queue.remote(chat_id)
        logger.info(f"스트리밍 처리 시작: chat_id={chat_id}")
        
        # 이미지 데이터 로그 출력 시 내용 생략
        http_query_print = http_query.copy()
        if "image_data" in http_query_print:
            http_query_print["image_data"] = "<omitted>"
        logger.debug(f"요청 데이터: {http_query_print}")

        # 비동기 처리를 위한 Future 생성
        loop = asyncio.get_event_loop()
        final_future = loop.create_future()
        
        queued_item = {
            "request_id": chat_id,
            "http_query": http_query,
        }

        await self.request_queue.put((queued_item, final_future))
        logger.info(f"처리 큐에 추가됨: chat_id={chat_id}")

        return chat_id
    
    # -------------------------------------------------------------------------
    # 대화 기록 및 참조 관련 메서드들
    # -------------------------------------------------------------------------
    async def get_conversation_history(self, request_id: str, last_index: int = None) -> dict:
        """
        요청 ID에 해당하는 대화 기록 조회
        
        Args:
            request_id: 요청 ID
            last_index: 마지막으로 받은 메시지 인덱스 (이후 메시지만 반환)
            
        Returns:
            dict: 직렬화된 대화 기록
        """
        try:
            if request_id in self.memory_map:
                memory = self.memory_map[request_id]
                history_obj = memory.load_memory_variables({})
                if "history" in history_obj and isinstance(history_obj["history"], list):
                    # 직렬화
                    serialized = [serialize_message(msg) for msg in history_obj["history"]]
                    logger.info(f"대화 기록 반환: 메시지 {len(serialized)}개")
                    
                    # last_index 이후의 메시지만 반환
                    if last_index is not None and isinstance(last_index, int):
                        serialized = serialized[last_index+1:]
                        
                    return {"history": serialized}
                else:
                    logger.warning("대화 기록이 리스트 형식이 아님")
                    return {"history": []}
            else:
                logger.info(f"해당 ID의 대화 기록 없음: {request_id}")
                return {"history": []}
        except Exception as e:
            logger.error(f"대화 기록 조회 오류: {e}")
            return {"history": []}
        
    async def get_reference_data(self, chunk_ids: list) -> list:
        """
        청크 ID 목록에 해당하는 참조 데이터 조회
        
        Args:
            chunk_ids: 청크 ID 목록
            
        Returns:
            list: 참조 데이터 목록
        """
        try:
            result = []
            data = self.data
            
            # 주어진 청크 ID에 해당하는 데이터 찾기
            for cid in chunk_ids:
                if cid in data["chunk_ids"]:
                    idx = data["chunk_ids"].index(cid)
                    record = {
                        "file_name": data["file_names"][idx],
                        "title": data["titles"][idx],
                        "text": data["texts_vis"][idx],
                        "date": str(data["times"][idx])
                    }
                    result.append(record)
            
            logger.info(f"참조 데이터 조회: {len(result)}/{len(chunk_ids)} 항목 찾음")
            return result
        except Exception as e:
            logger.error(f"참조 데이터 조회 오류: {e}")
            return []
        
    # -------------------------------------------------------------------------
    # 답변 생성 중도 정지
    # -------------------------------------------------------------------------
    async def stop_generation(self, request_id: str) -> str:
        """
        Stop an ongoing generation process for the specified request ID
        
        Args:
            request_id: The request ID to stop
            
        Returns:
            str: Status message
        """
        logger.info(f"Attempting to stop generation for request_id={request_id}")
        
        try:
            # Check if the request is in active generations
            if request_id in self.active_generations:
                # 1. Send stop signal to vLLM if using vLLM
                if hasattr(self.model, "is_vllm") and self.model.is_vllm:
                    try:
                        # vLLM has an abort method we can call
                        await self.model.abort(request_id=request_id)
                        logger.info(f"vLLM abort called for {request_id}")
                    except Exception as e:
                        logger.warning(f"Could not abort vLLM request: {e}")
                        
                # 2. Send final streaming token to client
                try:
                    # Send special token to indicate stopping
                    await self.queue_manager.put_token.remote(
                        request_id, 
                        json.dumps({
                            "type": "answer", 
                            "answer": "\n\n[Generation stopped by user]",
                            "continue": "E"  # Mark as final
                        })
                    )
                    # Send stream done token
                    await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")
                except Exception as e:
                    logger.error(f"Error sending stop tokens: {e}")
                
                # 3) 중단된 요청 완료 처리 
                self.metrics.finish_request(request_id, answer_text="[중단됨]")

                # 4) SSE 큐 삭제
                await self.queue_manager.delete_queue.remote(request_id)

                # 5) active_generations에서 제거
                self.active_generations.discard(request_id)
                    
                return "Generation stopped successfully"
            else:
                logger.warning(f"Request {request_id} not found in active generations")
                return "No active generation found for this request"
                
        except Exception as e:
            logger.error(f"Error stopping generation: {e}")
            return f"Error stopping generation: {str(e)}"
        
    # -------------------------------------------------------------------------
    # PROMPT TESTING - 프롬프트 테스트 기능
    # -------------------------------------------------------------------------
    async def test_prompt(self, system_prompt: str, user_text: str, file_data: Optional[Any] = None, file_type: Optional[str] = None, request_id: str = None) -> str:
        """
        새 프롬프트를 테스트합니다.
        
        Args:
            system_prompt: 테스트할 시스템 프롬프트
            user_text: 사용자 입력 텍스트
            file_data: 파일 데이터 (선택)
            file_type: 파일 타입 ('image' 또는 'pdf')
            request_id: 요청 ID
            
        Returns:
            str: 생성된 결과
        """
        logger.info(f"프롬프트 테스트 메서드 호출: prompt_length={len(system_prompt)}, file_type={file_type}, request_id={request_id}")
        
        result = await test_prompt_with_image(
            system_prompt=system_prompt,
            user_text=user_text,
            file_data=file_data,
            file_type=file_type,
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            request_id=request_id
        )
        
        return result

    async def test_prompt_stream(self, system_prompt: str, user_text: str, file_data: Optional[Any] = None, file_type: Optional[str] = None, request_id: str = None) -> str:
        """
        새 프롬프트를 스트리밍 방식으로 테스트합니다.
        
        Args:
            system_prompt: 테스트할 시스템 프롬프트
            user_text: 사용자 입력 텍스트
            file_data: 파일 데이터 (선택)
            file_type: 파일 타입 ('image' 또는 'pdf')
            request_id: 요청 ID
            
        Returns:
            str: 채팅 ID (스트리밍용)
        """
        logger.info(f"스트리밍 프롬프트 테스트 시작: prompt_length={len(system_prompt)}, file_type={file_type}, request_id={request_id}")
        
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # 비동기 제너레이터를 처리하기 위한 작업자
        async def process_stream():
            try:
                async for chunk in test_prompt_streaming(
                    system_prompt=system_prompt,
                    user_text=user_text,
                    file_data=file_data,
                    file_type=file_type,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    config=self.config,
                    request_id=request_id
                ):
                    # 응답 토큰 JSON 형식으로 포장하여 전송
                    answer_json = json.dumps({
                        "type": "answer",
                        "answer": chunk
                    }, ensure_ascii=False)
                    await self.queue_manager.put_token.remote(request_id, answer_json)
                
                # 스트리밍 완료 신호 전송
                await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")
                
            except Exception as e:
                err_msg = f"스트리밍 프롬프트 테스트 오류: {str(e)}"
                logger.error(err_msg)
                
                # 에러 메시지 전송
                error_token = json.dumps({"type": "error", "message": err_msg}, ensure_ascii=False)
                await self.queue_manager.put_token.remote(request_id, error_token)
                await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")
        
        # 비동기 작업 시작
        asyncio.create_task(process_stream())
        
        return request_id
    
    # InferenceActor 클래스에 추가할 메서드
    async def _sync_metrics_with_collector(self):
        """주기적으로 로컬 지표를 전역 수집기와 동기화합니다."""
        self.actor_id = f"actor_{id(self)}"  # 각 액터의 고유 ID
        
        while True:
            try:
                if self.metrics_collector is not None:
                    # 현재 상태를 가져와서 수집기에 전송
                    state = self.metrics.dump_state()
                    
                    # 완료된 요청을 수집기에 등록
                    for completed_req in state.get("recent_finished", []):
                        await self.metrics_collector.register_actor_request.remote(
                            self.actor_id, completed_req
                        )
                    
                    # 액터 상태 정보 등록
                    await self.metrics_collector.register_actor_stats.remote(
                        self.actor_id, state["global"]
                    )
            except Exception as e:
                logging.error(f"지표 동기화 실패: {e}")
            
            # 1분마다 동기화
            await asyncio.sleep(60)
            
    # InferenceActor 클래스에 추가할 안전한 지표 메서드
    def safe_metrics_call(self, method_name, *args, **kwargs):
        """
        에러가 전파되지 않도록 지표 메서드를 안전하게 호출합니다.
        
        Args:
            method_name (str): 호출할 MetricsManager 메서드 이름
            *args, **kwargs: 메서드에 전달할 인자
        """
        try:
            if hasattr(self.metrics, method_name):
                method = getattr(self.metrics, method_name)
                if callable(method):
                    return method(*args, **kwargs)
        except Exception as e:
            logging.error(f"metrics.{method_name} 호출 중 오류: {e}")
        return None
    
    # ──────────────────────────────────────────────────────────
    #   메트릭 스냅샷 반환 (Dashboard / API 용)
    # ──────────────────────────────────────────────────────────
    def get_metrics_snapshot(self):
        return self.metrics.dump_state()

    # 이 메서드를 사용하여 기존 코드의 직접 호출을 대체합니다
    # 예를 들어, 아래와 같이 수정:

    # 기존:
    # self.metrics.start_request(request_id, user_input, rag=use_rag, image=bool(image_data))

    # 수정:
    # self.safe_metrics_call('start_request', request_id, user_input, rag=use_rag, image=bool(image_data))

# -------------------------------------------------------------------------
# Ray Serve 배포를 위한 서비스 클래스
# -------------------------------------------------------------------------
@serve.deployment(name="inference", max_ongoing_requests=100)
class InferenceService:
    """
    Ray Serve를 통한 배포를 위한 서비스 클래스
    각 요청을 InferenceActor로 전달하고 결과를 반환함
    """
    def __init__(self, config):
        """
        InferenceService 초기화
        
        Args:
            config: 설정 객체
        """
        self.config = config
        # metrics_collector 설정
        try:
            metrics_collector = ray.get_actor("MetricsCollector")
        except ValueError:
            metrics_collector = None
            logging.warning("MetricsCollector 액터를 찾을 수 없습니다 - 전역 지표가 비활성화됩니다")
        
        # metrics_collector를 InferenceActor에 전달
        self.actor = InferenceActor.options(
            num_gpus=config.ray.num_gpus, 
            num_cpus=config.ray.num_cpus
        ).remote(config, metrics_collector)

    async def query(self, http_query: dict):
        """
        일반 쿼리 처리 API
        
        Args:
            http_query: 요청 데이터
            
        Returns:
            dict: 처리 결과
        """
        result = await self.actor.process_query.remote(http_query)
        return result
    
    async def process_query_stream(self, http_query: dict) -> str:
        """
        스트리밍 쿼리 처리 API
        
        Args:
            http_query: 요청 데이터
            
        Returns:
            str: 생성된 채팅/요청 ID
        """
        req_id = await self.actor.process_query_stream.remote(http_query)
        return req_id
    
    async def get_history(self, request_id: str, last_index: int = None):
        """
        대화 기록 조회 API
        
        Args:
            request_id: 요청 ID
            last_index: 마지막으로 받은 메시지 인덱스
            
        Returns:
            dict: 대화 기록
        """
        result = await self.actor.get_conversation_history.remote(request_id, last_index)
        return result

    async def get_reference_data(self, chunk_ids: list):
        """
        참조 데이터 조회 API
        
        Args:
            chunk_ids: 청크 ID 목록
            
        Returns:
            list: 참조 데이터 목록
        """
        result = await self.actor.get_reference_data.remote(chunk_ids)
        return result
    
    # Finally, add this to the InferenceService class
    async def stop_generation(self, request_id: str) -> str:
        """
        Stop generation API
        
        Args:
            request_id: The request ID to stop
            
        Returns:
            str: Status message
        """
        result = await self.actor.stop_generation.remote(request_id)
        return result
    
    # InferenceService 클래스에 다음 메서드 추가
    async def test_prompt(self, system_prompt: str, user_text: str, file_data: Optional[Any] = None, file_type: Optional[str] = None, request_id: str = None) -> str:
        """
        새 프롬프트 테스트 API
        
        Args:
            system_prompt: 테스트할 시스템 프롬프트
            user_text: 사용자 입력 텍스트
            file_data: 파일 데이터 (선택)
            file_type: 파일 타입 ('image' 또는 'pdf')
            request_id: 요청 ID
            
        Returns:
            str: 생성된 결과
        """
        result = await self.actor.test_prompt.remote(system_prompt, user_text, file_data, file_type, request_id)
        return result

    async def test_prompt_stream(self, system_prompt: str, user_text: str, file_data: Optional[Any] = None, file_type: Optional[str] = None, request_id: str = None) -> str:
        """
        새 프롬프트 스트리밍 테스트 API
        
        Args:
            system_prompt: 테스트할 시스템 프롬프트
            user_text: 사용자 입력 텍스트
            file_data: 파일 데이터 (선택)
            file_type: 파일 타입 ('image' 또는 'pdf')
            request_id: 요청 ID
            
        Returns:
            str: 채팅 ID (스트리밍용)
        """
        req_id = await self.actor.test_prompt_stream.remote(system_prompt, user_text, file_data, file_type, request_id)
        return req_id
    
    async def metrics(self):
        """
        GET /metrics 에서 호출 – Actor 의 메트릭 스냅샷 반환
        """
        return await self.actor.get_metrics_snapshot.remote()
