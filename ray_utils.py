# ray_utils.py
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

from RAG import (
    query_sort,
    execute_rag,
    generate_answer,
    generate_answer_stream,
)  # hypothetically
from utils import (
    load_model,
    load_data,
    process_format_to_response,
    process_to_format,
    error_format,
)
# from summarizer import summarize_conversation
from summarizer import summarize_conversation
from debug_tracking import log_batch_info, log_system_info

# 랭체인 도입
from langchain.memory import ConversationBufferMemory

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
        # 비동기 큐와 배치 처리 설정 (마이크로배칭)
        self.request_queue = asyncio.Queue()
        self.max_batch_size = config.ray.max_batch_size  # 최대 배치 수
        self.batch_wait_timeout = config.ray.batch_wait_timeout  # 배치당 처리 시간

        # Actor 내부에서 ProcessPoolExecutor 생성 (직렬화 문제 회피)
        max_workers = int(min(config.ray.num_cpus * 0.8, (26*config.ray.actor_count)-4))
        self.process_pool = ProcessPoolExecutor(max_workers)

        self.queue_manager = ray.get_actor("SSEQueueManager")
        # --- NEW OR MODIFIED ---
        # A dictionary to store SSE queues for streaming requests
        # Key = request_id, Value = an asyncio.Queue of partial token strings
        self.active_sse_queues: Dict[str, asyncio.Queue] = {}

        self.batch_counter = 0  # New counter to track batches

        # Micro-batching만 적용
        # asyncio.create_task(self._batch_processor())

        # ---------------------------
        # LangChain Memory 맵 (랭체인)
        # key: request_id, value: ConversationBufferMemory()
        # ---------------------------
        self.memory_map = {}

        # In-flight batching까지 추가 적용
        asyncio.create_task(self._in_flight_batch_processor())


    def get_memory_for_session(self, request_id: str) -> ConversationBufferMemory:
        """
        세션별 Memory를 안전하게 가져오는 헬퍼 메서드.
        만약 memory_map에 request_id가 없으면 새로 생성해 저장 후 반환.
        """
        if request_id not in self.memory_map:
            print(f"[DEBUG] Creating new ConversationBufferMemory for session={request_id}")
            self.memory_map[request_id] = ConversationBufferMemory(return_messages=True)
        return self.memory_map[request_id]


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

    # -------------------------------------------------------------------------
    # Micro_batch_processor
    # -------------------------------------------------------------------------

    async def _batch_processor(self):
        """
        Continuously processes queued requests in batches (micro-batching).
        We add new logic for streaming partial tokens if a request has an SSE queue.
        """
        while True:
            batch = []
            batch_start_time = time.time()
            # 1) get first request from the queue
            print("=== _batch_processor waiting for request_queue item... ===")
            item = await self.request_queue.get()
            print(
                f"[DEBUG] 첫 요청 도착: {time.strftime('%H:%M:%S')} (현재 배치 크기: 1)"
            )
            batch.append(item)

            print(f"[DEBUG] Received first request at {time.strftime('%H:%M:%S')}")

            # 2) try to fill the batch up to batch_size or until timeout
            try:
                while len(batch) < self.max_batch_size:
                    print("현재 배치 사이즈 : ", len(batch))
                    print("최대 배치 사이즈 : ", self.max_batch_size)
                    item = await asyncio.wait_for(
                        self.request_queue.get(), timeout=self.batch_wait_timeout
                    )

                    batch.append(item)
                    print(
                        f"[DEBUG] 추가 요청 도착: {time.strftime('%H:%M:%S')} (현재 배치 크기: {len(batch)})"
                    )
            except asyncio.TimeoutError:
                elapsed = time.time() - batch_start_time
                print(
                    f"[DEBUG] 타임아웃 도달: {elapsed:.2f}초 후 (최종 배치 크기: {len(batch)})"
                )
                pass

            print(
                f"=== _batch_processor: 배치 사이즈 {len(batch)} 처리 시작 ({time.strftime('%H:%M:%S')}) ==="
            )

            # 각 요청 처리 전후에 로그 추가
            start_proc = time.time()
            await asyncio.gather(
                *(
                    self._process_single_query(req, fut, sse_queue)
                    for (req, fut, sse_queue) in batch
                )
            )
            proc_time = time.time() - start_proc
            print(f"[DEBUG] 해당 배치 처리 완료 (처리시간: {proc_time:.2f}초)")

    # -------------------------------------------------------------------------
    # In-flight BATCH PROCESSOR
    # -------------------------------------------------------------------------

    async def _in_flight_batch_processor(self):
        while True:
            # Wait for the first item (blocking until at least one is available)
            print(
                "=== [In-Flight Batching] Waiting for first item in request_queue... ==="
            )
            first_item = await self.request_queue.get()
            batch = [first_item]
            batch_start_time = time.time()

            print(
                "[In-Flight Batching] Got the first request. Attempting to fill a batch..."
            )

            # Attempt to fill up the batch until we hit max_batch_size or batch_wait_timeout
            while len(batch) < self.max_batch_size:
                try:
                    remain_time = self.batch_wait_timeout - (
                        time.time() - batch_start_time
                    )
                    if remain_time <= 0:
                        print(
                            "[In-Flight Batching] Timed out waiting for more requests; proceeding with current batch."
                        )
                        break
                    item = await asyncio.wait_for(
                        self.request_queue.get(), timeout=remain_time
                    )
                    batch.append(item)
                    print(
                        f"[In-Flight Batching] +1 request => batch size now {len(batch)} <<< {self.max_batch_size}"
                    )
                except asyncio.TimeoutError:
                    print(
                        "[In-Flight Batching] Timeout reached => proceeding with the batch."
                    )
                    break
            self.batch_counter += 1
            
            # 현재 배치 정보 로깅
            log_batch_info(batch)
            log_system_info("배치 처리 전 상태")

            # We have a batch of items: each item is ( http_query_or_stream_dict, future, sse_queue )
            # We'll process them concurrently.
            tasks = []
            for request_tuple in batch:
                request_obj, fut, sse_queue = request_tuple
                tasks.append(self._process_single_query(request_obj, fut, sse_queue))

            # Actually run them all concurrently
            await asyncio.gather(*tasks)
            log_system_info("배치 처리 후 상태")

    async def _process_single_query(self, http_query_or_stream_dict, future, sse_queue):
        """
        Process a single query from the micro-batch. If 'sse_queue' is given,
        we do partial-token streaming. Otherwise, normal final result.
        """
        # 스트리밍 요청인 경우 request_id를 미리 초기화
        request_id = None
        print(
            f"[DEBUG] _process_single_query 시작: {time.strftime('%H:%M:%S')}, 요청 내용: {http_query_or_stream_dict}, 현재 스레드: {threading.current_thread().name}"
        )
        try:
            # 1) request_id 구분
            if (
                isinstance(http_query_or_stream_dict, dict)
                and "request_id" in http_query_or_stream_dict
            ):
                # It's a streaming request
                request_id = http_query_or_stream_dict["request_id"]
                http_query = http_query_or_stream_dict["http_query"]
                is_streaming = True
                print(f"[STREAM] _process_single_query: request_id={request_id}")
            else:
                # It's a normal synchronous request
                request_id = None
                http_query = http_query_or_stream_dict
                is_streaming = False
                print("[SYNC] _process_single_query started...")
                
            # 2) Memory 객체 가져오기 (없으면 새로 생성)
            if request_id not in self.memory_map:
                self.memory_map[request_id] = ConversationBufferMemory(return_messages=True)
            
            memory = self.memory_map[request_id]

            # 3) 유저가 현재 입력한 쿼리 가져오기
            user_input = http_query.get("qry_contents", "")
            
            # 4) LangChain Memory에서 이전 대화 이력(history) 추출
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
            
            # # 2) 추가: 전체 토큰 수가 4000개를 초과하면 마지막 4000 토큰만 유지
            # past_tokens = self.tokenizer.tokenize(str(past_context))
            # if len(past_tokens) > 4000:
            #     past_tokens = past_tokens[-4000:]
            #     past_context = self.tokenizer.convert_tokens_to_string(past_tokens)
            
            # ★ 토큰 수 계산 코드 추가 ★
            # retrieval 자료는 dict나 리스트일 수 있으므로 문자열로 변환하여 토큰화합니다.
            # 각 입력값을 명시적으로 str()로 변환합니다.
            past_tokens = self.tokenizer.tokenize(str(past_context))
            query_tokens = self.tokenizer.tokenize(str(user_input))
            total_tokens = len(past_tokens) + len(query_tokens)
            print(f"[DEBUG] Token counts - 이전 대화: {len(past_tokens)}, 사용자 입력 질문: {len(query_tokens)}, 총합: {total_tokens}")
            
            # # To Calculate the token
            # tokens = self.tokenizer(user_input, add_special_tokens=True)["input_ids"]
            # print(f"[DEBUG] Processing query: '{user_input}' with {len(tokens)} tokens")

            # 5) 필요하다면 데이터를 다시 로드(1.16version 유지)
            self.data = load_data(
                self.config.data_path
            )  # if you want always-latest, else skip

            # 6) 현재 사용중인 Thread 확인
            print("   ... calling query_sort() ...")
            # print(
            #     f"[DEBUG] query_sort 시작 (offload) - 스레드: {threading.current_thread().name}"
            # )
            # 7) “대화 이력 + 현재 사용자 질문”을 Prompt에 합쳐서 RAG 수행
            #    방법 1) query_sort() 전에 past_context를 참조해 query를 확장
            #    방법 2) generate_answer()에서 Prompt 앞부분에 붙임
            # 여기서는 예시로 “query_sort”에 past_context를 넘겨
            # 호출부 수정
            params = {
                "user_input": f"{past_context}\n사용자 질문: {user_input}",
                "model": self.model,
                "tokenizer": self.tokenizer,
                "embed_model": self.embed_model,
                "embed_tokenizer": self.embed_tokenizer,
                "data": self.data,
                "config": self.config,
            }
            QU, KE, TA, TI = await query_sort(params)
            print(f"   ... query_sort => QU={QU}, KE={KE}, TA={TA}, TI={TI}")

            # 4) RAG
            if TA == "yes":
                try:
                    docs, docs_list = execute_rag(
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
                        retrieval, chart = process_to_format(docs_list, type="SQL")
                    except Exception as e:
                        print("[ERROR] process_to_format (SQL) failed:", str(e))
                        retrieval, chart = [], None

                    # If streaming => partial tokens
                    if is_streaming:
                        print(
                            f"[STREAM] Starting partial generation for request_id={request_id}"
                        )
                        await self._stream_partial_answer(
                            QU, docs, retrieval, chart, request_id, future, user_input
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
                        
                        # >>> CHANGED: Record used chunk IDs and summarize the conversation
                        
                        chunk_ids_used = []
                        for doc in docs_list:
                            if "chunk_id" in doc:
                                chunk_ids_used.append(doc["chunk_id"])
                        loop = asyncio.get_event_loop()
                        prev_summary = memory.load_memory_variables({}).get("summary", "")
                        new_entry = f"User: {user_input}\nAssistant: {output}\nUsed Chunks: {chunk_ids_used}\n"
                        updated_conversation = prev_summary + "\n" + new_entry
                        # Summarized CPU 사용
                        # import concurrent.futures

                        # Create a dedicated pool with more workers (e.g., 4)
                        # summary_pool = concurrent.futures.ProcessPoolExecutor(max_workers=4)

                        # Later, when calling the summarization function:
                        summarized = loop.run_in_executor(None, summarize_conversation, updated_conversation)
                        # # After obtaining 'summarized' in _process_single_query:
                        # if not summarized:
                        #     print("[ERROR] Summarization returned an empty string.")
                        # else:
                        #     print(f"[CHECK] Summarized conversation: {summarized}")
                        memory.save_context({"input": user_input}, {"output": output, "chunk_ids": chunk_ids_used, "summary": summarized})
                        # >>> CHANGED -----------------------------------------------------
                        
                        future.set_result(outputs)

                except Exception as e:
                    outputs = error_format("내부 Excel 에 해당 자료가 없습니다.", 551)
                    future.set_result(outputs)

            else:
                try:
                    print("[SOOWAN] TA is No, before make a retrieval")
                    docs, docs_list = execute_rag(
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
                            QU, docs, retrieval, None, request_id, future, user_input
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
                        
                        # >>> CHANGED: Record used chunk IDs and update conversation summary
                        chunk_ids_used = []
                        for doc in docs_list:
                            if "chunk_id" in doc:
                                chunk_ids_used.append(doc["chunk_id"])
                        loop = asyncio.get_event_loop()
                        prev_summary = memory.load_memory_variables({}).get("summary", "")
                        new_entry = f"User: {user_input}\nAssistant: {output}\nUsed Chunks: {chunk_ids_used}\n"
                        updated_conversation = prev_summary + "\n" + new_entry
                        # # Summarized CPU 사용
                        # import concurrent.futures
                        
                        # # Create a dedicated pool with more workers (e.g., 4)
                        # summary_pool = concurrent.futures.ProcessPoolExecutor(max_workers=4)
                        
                        # Later, when calling the summarization function:
                        summarized = loop.run_in_executor(None, summarize_conversation, updated_conversation)
                        
                        memory.save_context({"input": user_input}, {"output": output, "chunk_ids": chunk_ids_used, "summary": summarized})
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
        self, QU, docs, retrieval, chart, request_id, future, user_input
    ):
        """
        Instead of returning a final string, we generate partial tokens
        and push them to the SSE queue in real time.
        We'll do a "delta" approach so each chunk is only what's newly added.
        """
        print(
            f"[STREAM] _stream_partial_answer => request_id={request_id}, chart={chart}"
        )

        # 단일
        # queue = self.active_sse_queues.get(request_id)
        # if not queue:
        #     print(f"[STREAM] SSE queue not found => fallback to normal final (request_id={request_id})")
        #     # fallback...
        #     return

        # This will hold the entire text so far. We'll yield only new pieces.
        
        # 먼저, 참조 데이터 전송: type을 "reference"로 명시
        reference_json = json.dumps({
            "type": "reference",
            "status_code": 200,
            "result": "OK",
            "detail": "Reference data",
            "evt_time": datetime.datetime.now().isoformat(),
            "data_list": retrieval
        }, ensure_ascii=False)
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
            
            # # 2) 추가: 전체 토큰 수가 4000개를 초과하면 마지막 4000 토큰만 유지
            # past_tokens = self.tokenizer.tokenize(str(past_context))
            # if len(past_tokens) > 4000:
            #     past_tokens = past_tokens[-4000:]
            #     past_context = self.tokenizer.convert_tokens_to_string(past_tokens)
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
                final_query, docs, self.model, self.tokenizer, self.config
            ):
                # print(f"[STREAM] Received partial_text: {partial_text}")
                new_text = partial_text[len(partial_accumulator) :]
                partial_accumulator = partial_text
                if not new_text.strip():
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
            # 이제 memory에 저장 (이미 request_id를 알고 있다고 가정) # 랭체인
            try:
                memory.save_context({"input": user_input}, {"output": final_text})
            except Exception as e:
                msg = f"[STREAM] memory.save_context failed: {str(e)}"
                print(msg)

                
            if chart is not None:
                ans = process_to_format([final_text, chart], type="Answer")
                final_res = process_format_to_response(retrieval, ans)
            else:
                ans = process_to_format([final_text], type="Answer")
                final_res = process_format_to_response(retrieval, ans)
            # >>> CHANGED: Update conversation summary in streaming branch as well
            chunk_ids_used = []
            for doc in retrieval:
                if isinstance(doc, dict) and "chunk_id" in doc:
                    chunk_ids_used.append(doc["chunk_id"])
            loop = asyncio.get_event_loop() # CHANGED
            prev_summary = memory.load_memory_variables({}).get("summary", "")
            new_entry = f"User: {user_input}\nAssistant: {final_text}\nUsed Chunks: {chunk_ids_used}\n"
            updated_conversation = prev_summary + "\n" + new_entry
            # Inside _process_single_query, after getting the summarized text:
            # # Summarized CPU 사용
            # import concurrent.futures

            # # Create a dedicated pool with more workers (e.g., 4)
            # summary_pool = concurrent.futures.ProcessPoolExecutor(max_workers=4)

            # # Later, when calling the summarization function:
            summarized = loop.run_in_executor(None, summarize_conversation, updated_conversation)
            # if not summarized:
            #     print("[ERROR] Summarization returned an empty string.")
            # else:
            #     print("[CHECK] Summarized conversation:", summarized)
            
            memory.save_context({"input": user_input}, {"output": final_text, "chunk_ids": chunk_ids_used, "summary": summarized})
            # >>> CHANGED: -------------------------------------------------------
            future.set_result(final_res)
            await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")
            print(
                f"[STREAM] done => placed [[STREAM_DONE]] for request_id={request_id}"
            )
        except Exception as e:
            msg = f"[STREAM] error in partial streaming => {str(e)}"
            future.set_result(error_format(msg, 500))
            await self.queue_manager.put_token.remote(request_id, "[[STREAM_DONE]]")

    # ------------------------------------------------------------
    # NEW METHODS TO SUPPORT SSE
    # ------------------------------------------------------------
    # ----------------------
    # 1) Streaming Entrypoint
    # ----------------------
    async def process_query_stream(self, http_query: dict) -> str:
        """
        Called from /query_stream route.
        Create request_id, SSE queue, push to the micro-batch, return request_id.
        """
        # 사용자로부터 Request_id를 받거나 그렇지 않은 경우, 이를 랜덤으로 생성
        request_id = http_query.get("request_id")
        if not request_id:
            request_id = str(uuid.uuid4())
        await self.queue_manager.create_queue.remote(request_id)
        print(f"[STREAM] process_query_stream => request_id={request_id}, http_query={http_query}")


        loop = asyncio.get_event_loop()
        final_future = loop.create_future()

        sse_queue = asyncio.Queue()
        self.active_sse_queues[request_id] = sse_queue
        print(f"[STREAM] Created SSE queue for request_id={request_id}")

        # We'll push a special item (dict) onto the micro-batch queue
        queued_item = {
            "request_id": request_id,
            "http_query": http_query,
        }

        print(f"[STREAM] Putting item into request_queue for request_id={request_id}")
        await self.request_queue.put((queued_item, final_future, sse_queue))
        print(f"[STREAM] Done putting item in queue => request_id={request_id}")

        return request_id

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


# Too using about two actor


# Ray Serve를 통한 배포
@serve.deployment(
    name="inference",
    max_ongoing_requests=50,
    )
class InferenceService:
    def __init__(self, config):
        self.config = config
        self.actor = InferenceActor.options(
            num_gpus=config.ray.num_gpus, num_cpus=config.ray.num_cpus
        ).remote(config)

    async def query(self, http_query: dict):
        result = await self.actor.process_query.remote(http_query)
        return result

    async def process_query_stream(self, http_query: dict) -> str:
        req_id = await self.actor.process_query_stream.remote(http_query)
        return req_id

    async def pop_sse_token(self, req_id: str) -> str:
        token = await self.actor.pop_sse_token.remote(req_id)
        return token

    async def close_sse_queue(self, req_id: str) -> str:
        await self.actor.close_sse_queue.remote(req_id)
        return "closed"


# Ray의 요청을 비동기적으로 관리하기 위해 도입하는 큐-매니저
@ray.remote
class SSEQueueManager:
    def __init__(self):
        self.active_queues = {}
        self.lock = asyncio.Lock()

    async def create_queue(self, request_id):
        async with self.lock:
            self.active_queues[request_id] = asyncio.Queue()
            return True

    async def get_queue(self, request_id):
        return self.active_queues.get(request_id)

    async def get_token(self, request_id, timeout: float):
        queue = self.active_queues.get(request_id)
        if queue:
            try:
                token = await asyncio.wait_for(queue.get(), timeout=timeout)
                return token
            except asyncio.TimeoutError:
                return None
        return None

    async def put_token(self, request_id, token):
        async with self.lock:
            if request_id in self.active_queues:
                await self.active_queues[request_id].put(token)
                return True
            return False

    async def delete_queue(self, request_id):
        async with self.lock:
            if request_id in self.active_queues:
                del self.active_queues[request_id]
                return True
            return False
