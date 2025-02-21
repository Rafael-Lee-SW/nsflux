# ray_utils.py
import ray  # Ray library
from ray import serve
import json
import asyncio  # async I/O process module
import uuid  # --- NEW OR MODIFIED ---
import time
from typing import Dict, Optional  # --- NEW OR MODIFIED ---

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
        self.max_batch_size = 10  # 최대 배치 수
        self.batch_wait_timeout = 5  # 배치당 처리 시간

        # --- NEW OR MODIFIED ---
        # A dictionary to store SSE queues for streaming requests
        # Key = request_id, Value = an asyncio.Queue of partial token strings
        self.active_sse_queues: Dict[str, asyncio.Queue] = {}

        # 끝: start the micro-batch processor
        asyncio.create_task(self._batch_processor())

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
        print("self.request_queue : ", self.request_queue)
        return await future

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
            print("현재 request_queue 상태 : ", self.request_queue)
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
                    print(f"[DEBUG] Received additional request at {time.strftime('%H:%M:%S')}; batch size now: {len(batch)}")
            except asyncio.TimeoutError:
                print(f"[DEBUG] Timeout reached after {time.time() - batch_start_time:.2f} seconds; proceeding with batch size: {len(batch)}")
                pass

            print(f"=== _batch_processor got a batch of size {len(batch)} ===")

            # Now we have a batch of items: (http_query or {streaming}, future, sse_queue)
            # We'll process them concurrently, same as your existing code.
            # But if sse_queue is not None, that means "streaming request."

            # We'll keep your same approach: for each item, call _process_single_query,
            # except we add partial-token streaming inside that method if needed.
            
            # --- Calculate and log total input tokens in this batch ---
            total_tokens = 0
            for idx, (req, fut, sse_queue) in enumerate(batch):
                # For streaming requests, the query is under "qry_contents"
                if isinstance(req, dict):
                    query_text = req.get("qry_contents", "")
                else:
                    query_text = ""
                if query_text:
                    tokens = self.tokenizer(query_text, add_special_tokens=True)["input_ids"]
                    token_count = len(tokens)
                    total_tokens += token_count
                    print(f"[DEBUG] Batch item {idx+1}: query: '{query_text}' uses {token_count} tokens")
                else:
                    print(f"[DEBUG] Batch item {idx+1}: No query text found.")
            print(f"[BATCH] Total input tokens in current batch: {total_tokens}")
            
            await asyncio.gather(
                *(
                    self._process_single_query(req, fut, sse_queue)
                    for (req, fut, sse_queue) in batch
                )
            )

    async def _process_single_query(self, http_query_or_stream_dict, future, sse_queue):
        """
        Process a single query from the micro-batch. If 'sse_queue' is given,
        we do partial-token streaming. Otherwise, normal final result.
        """
        try:
            # --- NEW OR MODIFIED ---
            # Distinguish between normal requests vs streaming requests:
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

            # 1) get user query
            user_input = http_query.get("qry_contents", "")
            # To Calculate the token
            tokens = self.tokenizer(user_input, add_special_tokens=True)["input_ids"]
            print(f"[DEBUG] Processing query: '{user_input}' with {len(tokens)} tokens")
            # 2) optionally reload data if needed
            self.data = load_data(
                self.config.data_path
            )  # if you want always-latest, else skip
            # 3) classify
            print("   ... calling query_sort() ...")
            QU, KE, TA, TI = await query_sort(
                user_input,
                model=self.model,
                tokenizer=self.tokenizer,
                embed_model=self.embed_model,
                embed_tokenizer=self.embed_tokenizer,
                data=self.data,
                config=self.config,
            )
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
                    retrieval, chart = process_to_format(docs_list, type="SQL")

                    # If streaming => partial tokens
                    if is_streaming:
                        print(
                            f"[STREAM] Starting partial generation for request_id={request_id}"
                        )
                        await self._stream_partial_answer(
                            QU, docs, retrieval, chart, request_id, future
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
                        outputs = process_format_to_response(retrieval, answer)
                        future.set_result(outputs)

                except Exception as e:
                    outputs = error_format("내부 Excel 에 해당 자료가 없습니다.", 551)
                    future.set_result(outputs)

            else:
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
                    retrieval = process_to_format(docs_list, type="Retrieval")

                    if is_streaming:
                        print(
                            f"[STREAM] Starting partial generation for request_id={request_id}"
                        )
                        await self._stream_partial_answer(
                            QU, docs, retrieval, None, request_id, future
                        )
                    else:
                        output = await generate_answer(
                            QU,
                            docs,
                            model=self.model,
                            tokenizer=self.tokenizer,
                            config=self.config,
                        )
                        print("process_to_format 이후에 OUTPUT:", output)
                        answer = process_to_format([output], type="Answer")
                        print("process_to_format 이후에 ANSWER:", answer)
                        outputs = process_format_to_response(retrieval, answer)
                        future.set_result(outputs)

                except Exception as e:
                    outputs = error_format("내부 PPT에 해당 자료가 없습니다.", 552)
                    future.set_result(outputs)

        except Exception as e:
            # If error, set the future
            err_msg = f"처리 중 오류 발생: {str(e)}"
            print("[ERROR]", err_msg)
            future.set_result(error_format(err_msg, 500))

    # ------------------------------------------------------------
    # NEW HELPER FOR STREAMING PARTIAL ANSWERS
    # ------------------------------------------------------------
    async def _stream_partial_answer(
        self, QU, docs, retrieval, chart, request_id, future
    ):
        """
        Instead of returning a final string, we generate partial tokens
        and push them to the SSE queue in real time.
        We'll do a "delta" approach so each chunk is only what's newly added.
        """
        print(f"[STREAM] _stream_partial_answer => request_id={request_id}, chart={chart}")

        queue = self.active_sse_queues.get(request_id)
        if not queue:
            print(f"[STREAM] SSE queue not found => fallback to normal final (request_id={request_id})")
            # fallback...
            return

        # This will hold the entire text so far. We'll yield only new pieces.
        partial_accumulator = ""

        try:
            print(f"[STREAM] SSE queue found => calling generate_answer_stream (request_id={request_id})")

            async for partial_text in generate_answer_stream(
                QU, docs, self.model, self.tokenizer, self.config
            ):
                # partial_text is the entire text so far. Let's get the new substring.
                new_text = partial_text[len(partial_accumulator):]
                partial_accumulator = partial_text  # update

                # Optionally skip if blank
                if not new_text.strip():
                    continue

                # print(f"[STREAM] yield new_text => '{new_text[:50]}...'") # Too large logs
                await queue.put(new_text)

            # Now partial_accumulator is the final text
            final_text = partial_accumulator
            # Build final JSON for the future
            if chart is not None:
                ans = process_to_format([final_text, chart], type="Answer")
                final_res = process_format_to_response(retrieval, ans)
            else:
                ans = process_to_format([final_text], type="Answer")
                final_res = process_format_to_response(retrieval, ans)

            future.set_result(final_res)
            await queue.put("[[STREAM_DONE]]")
            print(f"[STREAM] done => placed [[STREAM_DONE]] => request_id={request_id}")

        except Exception as e:
            msg = f"[STREAM] error in partial streaming => {str(e)}"
            print(msg)
            future.set_result(error_format(msg, 500))
            if queue:
                await queue.put("[[STREAM_DONE]]")

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
        request_id = str(uuid.uuid4())
        print(
            f"[STREAM] process_query_stream => request_id={request_id}, http_query={http_query}"
        )

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

# # Ray Serve를 통한 배포
# @serve.deployment(name="inference", num_replicas=1)
# class InferenceService:
#     def __init__(self, config):
#         self.config = config
#         self.actor = InferenceActor.options(num_gpus=config.ray.actor_num_gpus).remote(config)

#     async def query(self, http_query: dict):
#         result = await self.actor.process_query.remote(http_query)
#         return result

#     async def process_query_stream(self, http_query: dict) -> str:
#         req_id = await self.actor.process_query_stream.remote(http_query)
#         return req_id

#     async def pop_sse_token(self, req_id: str) -> str:
#         token = await self.actor.pop_sse_token.remote(req_id)
#         return token

#     async def close_sse_queue(self, req_id: str) -> str:
#         await self.actor.close_sse_queue.remote(req_id)
#         return "closed"