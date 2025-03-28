
# ray_deploy/ray_utils.py
import ray  # Ray library
from ray import serve
import json
import asyncio  # async I/O process module
from concurrent.futures import ProcessPoolExecutor  # 스레드 컨트롤
import uuid  # --- NEW OR MODIFIED ---
import time
from typing import Dict, Optional  # --- NEW OR MODIFIED ---
import threading  # To find out the usage of thread
import datetime

from core.RAG import (
    query_sort,
    specific_question,
    execute_rag,
    generate_answer,
    generate_answer_stream,
    image_query,
)  # hypothetically
from utils import (
    load_model,
    load_data,
    process_format_to_response,
    process_to_format,
    error_format,
)
# from summarizer import summarize_conversation
from utils.summarizer import summarize_conversation
from utils.debug_tracking import log_batch_info, log_system_info
# Langchain Memory system
from ray_deploy.langchain import CustomConversationBufferMemory, serialize_message

# Configuration
import yaml
from box import Box
with open("./config.yaml", "r") as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)

@ray.remote  # From Decorator, Each Actor is allocated 1 GPU
class InferenceActor:
    async def __init__(self, config):
        self.config = config
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
        # Key = request_id, Value = an asyncio.Queue of partial token strings
        # A dictionary to store SSE queues for streaming requests
        self.queue_manager = ray.get_actor("SSEQueueManager")
        self.active_sse_queues: Dict[str, asyncio.Queue] = {}

        self.batch_counter = 0  # New counter to track batches

        self.memory_map = {}
        
        # 활성 작업 추적을 위한 변수 추가
        self.active_tasks = set()
        self.max_concurrent_tasks = config.ray.max_batch_size
        
        # 연속 배치 처리기 시작 (Continuous batch)
        asyncio.create_task(self.continuous_batch_processor())
        
    # -------------------------------------------------------------------------
    # GET MEMORY FOR SESSION
    # -------------------------------------------------------------------------
    def get_memory_for_session(self, request_id: str) -> CustomConversationBufferMemory:
        """
        세션별 Memory를 안전하게 가져오는 헬퍼 메서드.
        만약 memory_map에 request_id가 없으면 새로 생성해 저장 후 반환.
        """
        if request_id not in self.memory_map:
            print(f"[DEBUG] Creating new CustomConversationBufferMemory for session={request_id}")
            self.memory_map[request_id] = CustomConversationBufferMemory(return_messages=True)
        return self.memory_map[request_id]
    # -------------------------------------------------------------------------
    # CONTINUOUOS BATCH PROCESSOR
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
                    # 새 요청 받기 (짧은 타임아웃으로 non-blocking 유지)
                    request_tuple = await asyncio.wait_for(
                        self.request_queue.get(), 
                        timeout=0.01
                    )
                    
                    # 비동기 처리 작업 생성
                    request_obj, fut, sse_queue = request_tuple
                    task = asyncio.create_task(
                        self._process_single_query(request_obj, fut, sse_queue)
                    )
                    
                    # 작업 완료 시 활성 목록에서 제거하는 콜백 설정
                    task.add_done_callback(
                        lambda t, task_ref=task: self.active_tasks.discard(task_ref)
                    )
                    
                    # 활성 작업 목록에 추가
                    self.active_tasks.add(task)
                    
                    print(f"[Continuous Batching] +1 request => active tasks now {len(self.active_tasks)}/{self.max_concurrent_tasks}")
                
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
                            print(f"[ERROR] Task failed: {e}")
                    
                    print(f"[Continuous Batching] Tasks completed: {len(done)} => active tasks now {len(self.active_tasks)}/{self.max_concurrent_tasks}")
                else:
                    await asyncio.sleep(0.01)

    # -------------------------------------------------------------------------
    # PROCESS SINGLE QUERY (TEXT TO TEXT)
    # -------------------------------------------------------------------------
    async def _process_single_query(self, http_query_or_stream_dict, future, sse_queue):
        """
        Process a single query from the micro-batch. If 'sse_queue' is given,
        we do partial-token streaming. Otherwise, normal final result.
        """
        body = http_query_or_stream_dict["http_query"]
        request_check = body.get("qry_contents", "")
        print(
            f"[DEBUG] _process_single_query 시작: {time.strftime('%H:%M:%S')}, 요청 내용: {request_check}, 현재 스레드: {threading.current_thread().name}"
        )
        
        # RAG on/off using boolean
        use_rag = body.get("use_rag")
        
        if use_rag is False:
            print(f"[NOT-USING-RAG] RAG is FALSE")
            
            # Determine) === Is streaming or not ===
            request_id = None
            # For streaming, the dict contains a "request_id"
            if isinstance(http_query_or_stream_dict, dict) and "request_id" in http_query_or_stream_dict:
                request_id = http_query_or_stream_dict["request_id"]
                http_query = http_query_or_stream_dict["http_query"]
                is_streaming = True
                print(f"[STREAM] _process_single_query: request_id={request_id}")
            else:
                request_id = None
                http_query = http_query_or_stream_dict
                is_streaming = False
                print("[NORMAL] _process_single_query started...")
                
            # 1) bring the user's input query
            user_input = http_query.get("qry_contents", "")
            print("[PROCESS_SINGLE_QUERY] user input : ", user_input)
            
            # Determine) === Make a description of image ===
            image_data = http_query.get("image_data")
                
            # 2) Memory 객체 정보 가져오기 (없으면 새로 생성)
            page_id = http_query.get("page_id", request_id)
            memory = self.get_memory_for_session(page_id)
            
            # 4) LangChain Memory에서 이전 대화 이력(history) 추출
            past_context = memory.load_memory_variables({}).get("history", [])
            # history가 리스트 형식인 경우 (각 메시지가 별도 항목으로 저장되어 있다면)
            if isinstance(past_context, list):
                recent_messages = [msg if isinstance(msg, str) else msg.content for msg in past_context[-5:]]
                past_context = "\n\n".join(recent_messages)
            else:
                # 문자열인 경우, 메시지 구분자를 "\n\n"으로 가정하여 분리
                messages = str(past_context).split("\n\n")
                recent_messages = messages[-5:]
                past_context = "\n\n".join(recent_messages)
                
            docs = None
            retrieval = None
            chart = None
            
            await self._stream_partial_answer(user_input, docs, retrieval, chart, request_id, future, user_input, http_query)
            
        else:
            try:
                # Determine) === Is streaming or not ===
                request_id = None
                # For streaming, the dict contains a "request_id"
                if isinstance(http_query_or_stream_dict, dict) and "request_id" in http_query_or_stream_dict:
                    request_id = http_query_or_stream_dict["request_id"]
                    http_query = http_query_or_stream_dict["http_query"]
                    is_streaming = True
                    print(f"[STREAM] _process_single_query: request_id={request_id}")
                else:
                    request_id = None
                    http_query = http_query_or_stream_dict
                    is_streaming = False
                    print("[NORMAL] _process_single_query started...")
                    
                # 1) bring the user's input query
                user_input = http_query.get("qry_contents", "")
                print("[PROCESS_SINGLE_QUERY] user input : ", user_input)
                
                # Determine) === Make a description of image ===
                image_data = http_query.get("image_data")
                image_description = {"is_structured": False, "description": "이미지는 입력되지 않았습니다."}
                if image_data is not None:
                    print("[DEBUG] _process_single_query: image_data detected, initiating image_sorting process.")
                    image_description = await image_query(http_query, self.model, config) 
                    
                # 2) Memory 객체 정보 가져오기 (없으면 새로 생성)
                page_id = http_query.get("page_id", request_id)
                memory = self.get_memory_for_session(page_id)
                
                # 4) LangChain Memory에서 이전 대화 이력(history) 추출
                past_context = memory.load_memory_variables({}).get("history", [])
                # history가 리스트 형식인 경우 (각 메시지가 별도 항목으로 저장되어 있다면)
                if isinstance(past_context, list):
                    recent_messages = [msg if isinstance(msg, str) else msg.content for msg in past_context[-5:]]
                    past_context = "\n\n".join(recent_messages)
                else:
                    # 문자열인 경우, 메시지 구분자를 "\n\n"으로 가정하여 분리
                    messages = str(past_context).split("\n\n")
                    recent_messages = messages[-5:]
                    past_context = "\n\n".join(recent_messages)
                
                # ★ 토큰 수 계산 코드 추가 ★
                past_tokens = self.tokenizer.tokenize(str(past_context))
                query_tokens = self.tokenizer.tokenize(str(user_input))
                total_tokens = len(past_tokens) + len(query_tokens)
                print(f"[DEBUG] Token counts - 이전 대화: {len(past_tokens)}, 사용자 입력 질문: {len(query_tokens)}, 총합: {total_tokens}")
                
                # 5) 필요하다면 RAG 데이터를 다시 로드(1.16version 유지)
                self.data = load_data(self.config.data_path)  # if you want always-latest, else skip
                
                # 6) “대화 이력 + 현재 사용자 질문”을 Prompt에 합쳐서 RAG 수행
                params = {
                    "user_input": f"사용자 질문: {user_input} [이미지 설명: {image_description.get('description')}]",
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                    "embed_model": self.embed_model,
                    "embed_tokenizer": self.embed_tokenizer,
                    "data": self.data,
                    "config": self.config,
                }
                print("[PROCESS_SINGLE_QUERY]... calling query_sort() ...")
                QU, KE, TA, TI = await query_sort(params)
                print(f"   ... query_sort => QU={QU}, KE={KE}, TA={TA}, TI={TI}")

                # 4) RAG
                if TA == "yes":
                    try:
                        print("[SOOWAN] config 설정 : ", self.config)
                        docs, docs_list = await execute_rag(
                            QU,
                            KE,
                            TA,
                            TI,
                            model=self.model,
                            tokenizer=self.tokenizer,
                            embed_model=self.embed_model,
                            embed_tokenizer=self.embed_tokenizer,
                            data=self.data,
                            config=self.config,
                        )
                        try:
                                                    # 기존 방식
                            retrieval, chart = process_to_format(docs_list, type="SQL")
                            # 수정된 방식 - Talbe,Chart 없이 Answer Part에 SQL 결과 전송.
                            # retrieval_sql = process_to_format(docs, type="Answer")
                            # await self.queue_manager.put_token.remote(request_id, retrieval_sql)
                        except Exception as e:
                            print("[ERROR] process_to_format (SQL) failed:", str(e))
                            retrieval, chart = [], None

                        # If streaming => partial tokens
                        if is_streaming:
                            print(
                                f"[STREAM] Starting partial generation for request_id={request_id}"
                            )
                            await self._stream_partial_answer(
                                QU, docs, retrieval, chart, request_id, future, user_input, http_query
                            )
                        else:
                            # normal final result
                            output = await generate_answer(
                                QU,
                                docs,
                                model=self.model,
                                tokenizer=self.tokenizer,
                                config=self.config,
                            )
                            answer = process_to_format([output, chart], type="Answer")
                            final_data = [retrieval, answer]
                            outputs = process_format_to_response(final_data, qry_id=None, continue_="C")
                            
                            # >>> Record used chunk IDs
                            # 변경 후: retrieval 결과에서 추출
                            chunk_ids_used = []
                            print("---------------- chunk_id 찾기 : ", retrieval.get("rsp_data", []))
                            for doc in retrieval.get("rsp_data", []):
                                if "chunk_id" in doc:
                                    chunk_ids_used.append(doc["chunk_id"])
                                                            
                            # 메모리에 저장
                            try:
                                memory.save_context(
                                    {
                                        "qry_contents": user_input,
                                        "qry_id": http_query.get("qry_id"),
                                        "user_id": http_query.get("user_id"),
                                        "auth_class": http_query.get("auth_class"),
                                        "qry_time": http_query.get("qry_time")
                                    },
                                    {
                                        "output": output,
                                        "chunk_ids": chunk_ids_used
                                    }
                                )
                            except Exception as e:
                                print(f"[ERROR memory.save_context] {e}")
                            # >>> CHANGED -----------------------------------------------------
                            future.set_result(outputs)

                    except Exception as e:
                        outputs = error_format("내부 Excel 에 해당 자료가 없습니다.", 551)
                        future.set_result(outputs)

                else:
                    try:
                        print("[SOOWAN] TA is No, before make a retrieval")
                        QU, KE, TA, TI = await specific_question(params) # TA == no, so that have to remake the question based on history
                        
                        docs, docs_list = await execute_rag(
                            QU,
                            KE,
                            TA,
                            TI,
                            model=self.model,
                            tokenizer=self.tokenizer,
                            embed_model=self.embed_model,
                            embed_tokenizer=self.embed_tokenizer,
                            data=self.data,
                            config=self.config,
                        )
                        retrieval = process_to_format(docs_list, type="Retrieval")
                        print("[SOOWAN] TA is No, and make a retrieval is successed")
                        if is_streaming:
                            print(
                                f"[STREAM] Starting partial generation for request_id={request_id}"
                            )
                            await self._stream_partial_answer(
                                QU, docs, retrieval, None, request_id, future, user_input, http_query
                            )
                        else:
                            output = await generate_answer(
                                QU,
                                docs,
                                model=self.model,
                                tokenizer=self.tokenizer,
                                config=self.config,
                            )
                            print("process_to_format 이후에 OUTPUT 생성 완료")
                            answer = process_to_format([output], type="Answer")
                            print("process_to_format 이후에 ANSWER까지 생성 완료")
                            final_data = [retrieval, answer]
                            outputs = process_format_to_response(final_data, qry_id=None, continue_="C")
                            
                            # >>> CHANGED: Record used chunk ID
                            chunk_ids_used = []
                            print("---------------- chunk_id 찾기 : ", retrieval.get("rsp_data", []))
                            for doc in retrieval.get("rsp_data", []):
                                if "chunk_id" in doc:
                                    chunk_ids_used.append(doc["chunk_id"])
                                    
                            # 메모리 저장
                            try:
                                memory.save_context(
                                    {
                                        "qry_contents": user_input,
                                        "qry_id": http_query.get("qry_id"),
                                        "user_id": http_query.get("user_id"),
                                        "auth_class": http_query.get("auth_class"),
                                        "qry_time": http_query.get("qry_time")
                                    },
                                    {
                                        "output": output,
                                        "chunk_ids": chunk_ids_used
                                    }
                                )
                            except Exception as e:
                                print(f"[ERROR memory.save_context] {e}")
                            # --------------------------------------------------------------------
                            
                            future.set_result(outputs)

                    except Exception as e:
                        # ====== 이 부분에서 SSE를 즉시 닫고 스트리밍 종료 ======
                        err_msg = f"[ERROR] 처리 중 오류 발생: {str(e)}"
                        print(err_msg)

                        # SSE 전송 (error 이벤트)
                        if request_id:
                            try:
                                error_token = json.dumps({"type": "error", "message": err_msg}, ensure_ascii=False)
                                await self.queue_manager.put_token.remote(request_id, error_token)
                                # 스트리밍 종료
                                await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")
                            except Exception as e2:
                                print(f"[ERROR] SSE 전송 중 추가 예외 발생: {str(e2)}")
                            finally:
                                # SSEQueue 정리
                                await self.close_sse_queue(request_id)

                        # Future 응답도 에러로
                        future.set_result(error_format(str(e), 500))
                        return
                    
            except Exception as e:
                err_msg = f"[ERROR] 처리 중 오류 발생: {str(e)}"
                print("[ERROR]", err_msg)
                # SSE 스트리밍인 경우 error 토큰과 종료 토큰 전송
                if request_id:
                    try:
                        error_token = json.dumps({"type": "error", "message": err_msg}, ensure_ascii=False)
                        await self.queue_manager.put_token.remote(request_id, error_token)
                    except Exception as e2:
                        print(f"[ERROR] SSE 전송 중 추가 예외 발생: {str(e2)}")
                future.set_result(error_format(err_msg, 500))
            finally:
                # 스트리밍 요청인 경우 반드시 SSE 큐에 종료 토큰을 넣고 큐를 정리한다.
                if request_id:
                    try:
                        await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")
                    except Exception as ex:
                        print(f"[DEBUG] Error putting STREAM_DONE: {str(ex)}")
                    await self.close_sse_queue(request_id)
                
    # ------------------------------------------------------------
    # HELPER FOR STREAMING PARTIAL ANSWERS (Modified to send reference)
    # ------------------------------------------------------------
    async def _stream_partial_answer(
        self, QU, docs, retrieval, chart, request_id, future, user_input, http_query
    ):
        """
        Instead of returning a final string, we generate partial tokens
        and push them to the SSE queue in real time.
        We'll do a "delta" approach so each chunk is only what's newly added.
        """
        print(
            f"[STREAM] _stream_partial_answer => request_id={request_id}, chart={chart}"
        )
        
        # 먼저, 참조 데이터 전송: type을 "reference"로 명시
        reference_json = json.dumps({
            "type": "reference",
            "status_code": 200,
            "result": "OK",
            "detail": "Reference data",
            "evt_time": datetime.datetime.now().isoformat(),
            "data_list": [retrieval]
        }, ensure_ascii=False)
        # Debug: print the reference JSON before sending
        print(f"[DEBUG] Prepared reference data: {reference_json}")
        await self.queue_manager.put_token.remote(request_id, reference_json)
        
        print(f"[STREAM] Sent reference data for request_id={request_id}")
        
        # 1) 메모리 가져오기 (없으면 생성)
        try:
            memory = self.get_memory_for_session(request_id)
        except Exception as e:
            msg = f"[STREAM] Error retrieving memory for {request_id}: {str(e)}"
            print(msg)
            # 에러 응답을 SSE로 전송하고 종료
            error_token = json.dumps({"type":"error","message":msg}, ensure_ascii=False)
            await self.queue_manager.put_token.remote(request_id, error_token)
            await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")
            future.set_result(error_format(msg, 500))
            return
        
        # 2) 과거 대화 이력 로드
        try:
            past_context = memory.load_memory_variables({})["history"]
            # history가 리스트 형식인 경우 (각 메시지가 별도 항목으로 저장되어 있다면)
            if isinstance(past_context, list):
                recent_messages = [msg if isinstance(msg, str) else msg.content for msg in past_context[-5:]]
                past_context = "\n\n".join(recent_messages)
            else:
                # 문자열인 경우, 메시지 구분자를 "\n\n"으로 가정하여 분리
                messages = str(past_context).split("\n\n")
                recent_messages = messages[-5:]
                past_context = "\n\n".join(recent_messages)
            
        except KeyError:
            # 만약 "history" 키가 없으면 빈 문자열로 처리
            print(f"[STREAM] No 'history' in memory for {request_id}, using empty.")
            past_context = ""
        except Exception as e:
            msg = f"[STREAM] load_memory_variables error for {request_id}: {str(e)}"
            print(msg)
            error_token = json.dumps({"type":"error","message":msg}, ensure_ascii=False)
            await self.queue_manager.put_token.remote(request_id, error_token)
            await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")
            future.set_result(error_format(msg, 500))
            return

        # 3) 최종 프롬프트 구성
        final_query = f"{past_context}\n\n[사용자 질문]\n{QU}"
        print(f"[STREAM] final_query = \n{final_query}")
        
        # ★ 토큰 수 계산 코드 추가 ★
        # retrieval 자료는 dict나 리스트일 수 있으므로 문자열로 변환하여 토큰화합니다.
        retrieval_str = str(retrieval)
        # 각 입력값을 명시적으로 str()로 변환합니다.
        past_tokens = self.tokenizer.tokenize(str(past_context))
        query_tokens = self.tokenizer.tokenize(str(QU))
        retrieval_tokens = self.tokenizer.tokenize(retrieval_str)
        total_tokens = len(self.tokenizer.tokenize(str(final_query))) + len(retrieval_tokens)
        print(f"[DEBUG] Token counts - 이전 대화: {len(past_tokens)}, RAG 검색 자료: {len(retrieval_tokens)}, 사용자 구체화 질문: {len(query_tokens)}, 총합: {total_tokens}")
        
        partial_accumulator = ""

        try:
            print(
                f"[STREAM] SSE: calling generate_answer_stream for request_id={request_id}"
            )
            async for partial_text in generate_answer_stream(
                final_query, docs, self.model, self.tokenizer, self.config, http_query
            ):
                # print(f"[STREAM] Received partial_text: {partial_text}")
                new_text = partial_text[len(partial_accumulator) :]
                partial_accumulator = partial_text
                
                # 수정: new_text가 완전히 빈 문자열("")인 경우에만 건너뛰기
                if new_text == "":
                    continue
                
                # Wrap answer tokens in a JSON object with type "answer"
                answer_json = json.dumps({
                    "type": "answer",
                    "answer": new_text
                }, ensure_ascii=False)
                # Use the central SSEQueueManager to put tokens
                # print(f"[STREAM] Sending token: {answer_json}")
                await self.queue_manager.put_token.remote(request_id, answer_json)
            final_text = partial_accumulator
                
            # >>> CHANGED: Update conversation summary in streaming branch as well
            chunk_ids_used = []
            print("---------------- chunk_id 찾기 : ", retrieval.get("rsp_data", []))
            for doc in retrieval.get("rsp_data", []):
                if "chunk_id" in doc:
                    chunk_ids_used.append(doc["chunk_id"])
                    
            # 메모리 저장
            try:
                memory.save_context(
                    {
                        "qry_contents": user_input,
                        "qry_id": "",  # 필요한 경우 http_query에 있는 값을 넣음
                    },
                    {
                        "output": final_text,
                        "chunk_ids": chunk_ids_used
                    }
                )
            except Exception as e:
                print(f"[ERROR memory.save_context in stream] {e}")
            
            print("메시지 저장 직후 chunk_id 확인 : ", memory)

            # 최종 응답 구조
            if chart is not None:
                ans = process_to_format([final_text, chart], type="Answer")
                final_res = process_format_to_response(retrieval, ans)
            else:
                ans = process_to_format([final_text], type="Answer")
                final_res = process_format_to_response(retrieval, ans)
                
            # 담아서 보내기
            future.set_result(final_res)
            await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")
            print(
                f"[STREAM] done => placed [[STREAM_DONE]] for request_id={request_id}"
            )
        except Exception as e:
            msg = f"[STREAM] error in partial streaming => {str(e)}"
            print(msg)
            future.set_result(error_format(msg, 500))
            await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")

    # --------------------------------------------------------
    # EXISTING METHODS FOR NORMAL QUERIES (unchanged)
    # --------------------------------------------------------
    async def process_query(self, http_query):
        """
        Existing synchronous method. Returns final string/dict once done.
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        # There's no SSE queue for normal queries
        sse_queue = None
        await self.request_queue.put((http_query, future, sse_queue))
        # print("self.request_queue : ", self.request_queue)
        return await future
    # ----------------------
    # 1) Streaming Entrypoint
    # ----------------------
    async def process_query_stream(self, http_query: dict) -> str:
        """
        /query_stream 호출 시 page_id(채팅방 id)를 기반으로 SSE queue 생성하고,
        대화 저장에 활용할 수 있도록 합니다.
        """
        # page_id를 채팅방 id로 사용 (없으면 생성)
        chat_id = http_query.get("page_id")
        if not chat_id:
            chat_id = str(uuid.uuid4())
        http_query["page_id"] = chat_id  # 강제 할당
        await self.queue_manager.create_queue.remote(chat_id)
        print(f"[STREAM] process_query_stream => chat_id={chat_id}")
        
        # http_query 전체를 출력할 때 image_data 내용은 생략(요약 정보만 출력)
        http_query_print = http_query.copy()
        if "image_data" in http_query_print:
            http_query_print["image_data"] = "<omitted>"
        print(f"[DEBUG] Built http_query: {http_query_print}")

        loop = asyncio.get_event_loop()
        final_future = loop.create_future()

        sse_queue = asyncio.Queue()
        self.active_sse_queues[chat_id] = sse_queue
        print(f"[STREAM] Created SSE queue for chat_id={chat_id}")

        # 기존과 동일하게 micro-batch queue에 푸시 (http_query에 새 필드들이 포함됨)
        queued_item = {
            "request_id": chat_id,   # 내부적으로 page_id를 request_id처럼 사용
            "http_query": http_query,
        }

        print(f"[STREAM] Putting item into request_queue for chat_id={chat_id}")
        await self.request_queue.put((queued_item, final_future, sse_queue))
        print(f"[STREAM] Done putting item in queue => chat_id={chat_id}")

        return chat_id

    # ----------------------
    # 2) SSE token popping
    # ----------------------
    async def pop_sse_token(self, request_id: str) -> Optional[str]:
        """
        The SSE route calls this repeatedly to get partial tokens.
        If no token is available, we block up to 120s, else return None.
        """
        if request_id not in self.active_sse_queues:
            print(
                f"[STREAM] pop_sse_token => no SSE queue found for request_id={request_id}"
            )
            return None

        queue = self.active_sse_queues[request_id]
        try:
            token = await asyncio.wait_for(queue.get(), timeout=120.0)
            # print(f"[STREAM] pop_sse_token => got token from queue: {token}")
            return token
        except asyncio.TimeoutError:
            print(
                f"[STREAM] pop_sse_token => timed out waiting for token, request_id={request_id}"
            )
            return None

    # ----------------------
    # 3) SSE queue cleanup
    # ----------------------
    async def close_sse_queue(self, request_id: str):
        """
        Called by the SSE route after finishing.
        Remove the queue from memory.
        """
        if request_id in self.active_sse_queues:
            print(
                f"[STREAM] close_sse_queue => removing SSE queue for request_id={request_id}"
            )
            del self.active_sse_queues[request_id]
        else:
            print(f"[STREAM] close_sse_queue => no SSE queue found for {request_id}")
    
    # ----------------------
    # /history | 대화 기록 가져오기
    # ----------------------
    async def get_conversation_history(self, request_id: str) -> dict:
        """
        Returns the conversation history for the given request_id.
        The messages are serialized into a JSON-friendly format.
        """
        try:
            if request_id in self.memory_map:
                memory = self.memory_map[request_id]
                history_obj = memory.load_memory_variables({})
                if "history" in history_obj and isinstance(history_obj["history"], list):
                    # 직렬화
                    serialized = [serialize_message(msg) for msg in history_obj["history"]]
                    print("[HISTORY] 대화 기록 반환(직렬화) : ", serialized)
                    return {"history": serialized}
                else:
                    print("[HISTORY] 대화 기록 반환(직렬화X) : ", history_obj)
                    return {"history": []}
            else:
                return {"history": []}
        except Exception as e:
            print(f"[ERROR get_conversation_history] {e}")
            return {"history": []}
        
    # ----------------------
    # /reference | 해당 답변의 출처 가져오기
    # ----------------------
    async def get_reference_data(self, chunk_ids: list):
        try:
            result = []
            data = self.data
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
            return result
        except Exception as e:
            print(f"[ERROR get_reference_data] {e}")
            return []

# Ray Serve를 통한 배포
@serve.deployment(name="inference", max_ongoing_requests=100)
class InferenceService:
    def __init__(self, config):
        self.config = config
        self.actor = InferenceActor.options(
            num_gpus=config.ray.num_gpus, 
            num_cpus=config.ray.num_cpus
        ).remote(config)

    # Text
    async def query(self, http_query: dict):
        result = await self.actor.process_query.remote(http_query)
        return result
    # Text Stream
    async def process_query_stream(self, http_query: dict) -> str:
        req_id = await self.actor.process_query_stream.remote(http_query)
        return req_id
    
    async def pop_sse_token(self, req_id: str) -> str:
        token = await self.actor.pop_sse_token.remote(req_id)
        return token

    async def close_sse_queue(self, req_id: str) -> str:
        await self.actor.close_sse_queue.remote(req_id)
        return "closed"
    
    # /history
    async def get_history(self, request_id: str, last_index: int = None):
        result = await self.actor.get_conversation_history.remote(request_id)
        if last_index is not None and isinstance(result.get("history"), list):
            result["history"] = result["history"][last_index+1:]
        return result

    # /reference
    async def get_reference_data(self, chunk_ids: list):
        result = await self.actor.get_reference_data.remote(chunk_ids)
        return result

