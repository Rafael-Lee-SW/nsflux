# ray_utils.py

import ray  # Ray library
import json
import asyncio  # async I/O process module
import uuid  # --- NEW OR MODIFIED ---
from typing import Dict, Optional  # --- NEW OR MODIFIED ---

from RAG import query_sort, execute_rag, generate_answer, generate_answer_stream  # hypothetically
from utils import (
    load_model,
    load_data,
    process_format_to_response,
    process_to_format,
    error_format,
)

@ray.remote(num_gpus=1)  # From Decorator, Each Actor is allocated 1 GPU
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
        self.batch_size = 10  # 최대 배치 수
        self.batch_delay = 0.5  # 배치당 처리 시간

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
        return await future

    async def _batch_processor(self):
        """
        Continuously processes queued requests in batches (micro-batching).
        We add new logic for streaming partial tokens if a request has an SSE queue.
        """
        while True:
            batch = []
            # 1) get first request from the queue
            print("=== _batch_processor waiting for request_queue item... ===")
            item = await self.request_queue.get()
            batch.append(item)

            # 2) try to fill the batch up to batch_size or until timeout
            try:
                while len(batch) < self.batch_size:
                    item = await asyncio.wait_for(self.request_queue.get(), timeout=self.batch_delay)
                    batch.append(item)
            except asyncio.TimeoutError:
                pass

            print(f"=== _batch_processor got a batch of size {len(batch)} ===")

            # Now we have a batch of items: (http_query or {streaming}, future, sse_queue)
            # We'll process them concurrently, same as your existing code. 
            # But if sse_queue is not None, that means "streaming request."

            # We'll keep your same approach: for each item, call _process_single_query,
            # except we add partial-token streaming inside that method if needed.
            await asyncio.gather(*(self._process_single_query(req, fut, sse_queue)
                                   for (req, fut, sse_queue) in batch))

    async def _process_single_query(self, http_query_or_stream_dict, future, sse_queue):
        """
        Process a single query from the micro-batch. If 'sse_queue' is given,
        we do partial-token streaming. Otherwise, normal final result.
        """
        try:
            # --- NEW OR MODIFIED ---
            # Distinguish between normal requests vs streaming requests:
            if isinstance(http_query_or_stream_dict, dict) and "request_id" in http_query_or_stream_dict:
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
            # 2) optionally reload data if needed
            self.data = load_data(self.config.data_path)  # if you want always-latest, else skip
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
                        QU, KE, TA, TI,
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
                        print(f"[STREAM] Starting partial generation for request_id={request_id}")
                        await self._stream_partial_answer(
                            QU, docs, retrieval, chart, request_id, future
                        )
                    else:
                        # normal final result
                        output = await generate_answer(
                            QU, docs,
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
                        QU, KE, TA, TI,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        embed_model=self.embed_model,
                        embed_tokenizer=self.embed_tokenizer,
                        data=self.data,
                        config=self.config,
                    )
                    retrieval = process_to_format(docs_list, type="Retrieval")
                    
                    if is_streaming:
                        print(f"[STREAM] Starting partial generation for request_id={request_id}")
                        await self._stream_partial_answer(
                            QU, docs, retrieval, None, request_id, future
                        )
                    else:
                        output = await generate_answer(
                            QU, docs,
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
    async def _stream_partial_answer(self, QU, docs, retrieval, chart, request_id, future):
        """
        Instead of returning a final string, we generate partial tokens
        and push them to the SSE queue in real time. 
        Once done, we set the 'future' to a final JSON result if you like.
        """
        # We'll do a naive approach: call `generate_answer(...)` once, 
        # but internally we want partial chunks. 
        # For full partial tokens with vLLM microbatch, we need a custom approach:
        # either we do run_iter, or we monkey-patch your generate. 
        #
        # Below is a simpler approach if you had an async generator for partial tokens:
        # e.g. `generate_answer_stream(...)`.
        # If you haven't implemented that, you'd do the advanced approach with run_iter.
        # For demonstration, let's pretend we have `generate_answer_stream`.

        print(f"[STREAM] _stream_partial_answer => retrieval={retrieval}, chart={chart}")

        queue = self.active_sse_queues.get(request_id)
        if not queue:
            # No SSE queue => can't stream
            # just do normal final
            print("[STREAM] SSE queue not found. Fallback to final result.")
            output = await generate_answer(QU, docs, model=self.model, tokenizer=self.tokenizer, config=self.config)
            # build final
            if chart is not None:
                answer = process_to_format([output, chart], type="Answer")
                final_res = process_format_to_response(retrieval, answer)
            else:
                answer = process_to_format([output], type="Answer")
                final_res = process_format_to_response(retrieval, answer)
            future.set_result(final_res)
            return

        # If we do have a queue, push partial tokens
        print(f"[STREAM] Found SSE queue. Begin partial generation for request_id={request_id}")
        try:
            async for partial_text in generate_answer_stream(QU, docs, self.model, self.tokenizer, self.config):
                await queue.put(partial_text)
            # After we finish, build final result
            # For demonstration, let final_text be everything joined or last chunk
            final_text = "All partial tokens joined."  # example
            if chart is not None:
                ans = process_to_format([final_text, chart], type="Answer")
                final_res = process_format_to_response(retrieval, ans)
            else:
                ans = process_to_format([final_text], type="Answer")
                final_res = process_format_to_response(retrieval, ans)
            # set the final result in the future
            future.set_result(final_res)

            # Then push a “DONE” marker
            await queue.put("[[STREAM_DONE]]")

        except Exception as e:
            err = f"[STREAM] Error in partial generation: {str(e)}"
            print(err)
            future.set_result(error_format(err, 500))
            # signal done
            if queue:
                await queue.put("[[STREAM_DONE]]")


    # ------------------------------------------------------------
    # NEW METHODS TO SUPPORT SSE
    # ------------------------------------------------------------
    async def process_query_stream(self, http_query: dict) -> str:
        """
        Called from Flask for streaming requests. We create a request_id,
        put it in the microbatch queue with an SSE queue,
        and return the request_id so the Flask route can read partial tokens.
        """
        request_id = str(uuid.uuid4())
        print(f"[STREAM] process_query_stream => request_id={request_id}")

        loop = asyncio.get_event_loop()
        final_future = loop.create_future()

        sse_queue = asyncio.Queue()
        self.active_sse_queues[request_id] = sse_queue

        queued_item = {
            "request_id": request_id,
            "http_query": http_query,
        }
        await self.request_queue.put((queued_item, final_future, sse_queue))

        return request_id

    async def pop_sse_token(self, request_id: str) -> Optional[str]:
        """
        The Flask route calls this repeatedly to pop the next partial token.
        If queue is empty, we block. If no queue or timed out => returns None.
        """
        if request_id not in self.active_sse_queues:
            return None
        try:
            token = await asyncio.wait_for(self.active_sse_queues[request_id].get(), timeout=120.0)
            return token
        except asyncio.TimeoutError:
            return None

    async def close_sse_queue(self, request_id: str):
        """
        Called by the Flask route after it finishes reading SSE tokens.
        Clean up memory by removing the queue.
        """
        if request_id in self.active_sse_queues:
            print(f"[STREAM] close_sse_queue => {request_id}")
            del self.active_sse_queues[request_id]
