import ray
import json
import asyncio
from RAG import query_sort, execute_rag, generate_answer
from utils import (
    load_model,
    load_data,
    process_format_to_response,
    process_to_format,
    error_format,
)


@ray.remote(num_gpus=1)
class InferenceActor:
    async def __init__(self, config):
        self.config = config
        # 액터 내부에서 모델 및 토크나이저를 새로 로드 (GPU에 한 번만 로드)
        self.model, self.tokenizer, self.embed_model, self.embed_tokenizer = load_model(
            config
        )
        # 데이터는 캐시 파일을 통해 로드 (필요 시, 메모리에 올리거나, 별도의 저장소 사용 고려)
        self.data = load_data(config.data_path)
        # 비동기 큐와 배치 처리 설정 (마이크로배칭)
        self.request_queue = asyncio.Queue()
        self.batch_size = 4
        self.batch_delay = 0.05
        asyncio.create_task(self._batch_processor())

    async def process_query(self, http_query):
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self.request_queue.put((http_query, future))
        return await future

    async def _batch_processor(self):
        while True:
            batch = []
            item = await self.request_queue.get()
            batch.append(item)
            try:
                while len(batch) < self.batch_size:
                    item = await asyncio.wait_for(
                        self.request_queue.get(), timeout=self.batch_delay
                    )
                    batch.append(item)
            except asyncio.TimeoutError:
                pass
            for http_query, future in batch:
                try:
                    user_input = http_query.get("qry_contents", "")
                    # 필요 시 최신 데이터 재로드 (옵션)
                    self.data = load_data(self.config.data_path)
                    QU, KE, TA, TI = query_sort(
                        user_input,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        embed_model=self.embed_model,
                        embed_tokenizer=self.embed_tokenizer,
                        data=self.data,
                        config=self.config,
                    )
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
                            output = generate_answer(
                                QU,
                                docs,
                                model=self.model,
                                tokenizer=self.tokenizer,
                                config=self.config,
                            )
                            answer = process_to_format([output, chart], type="Answer")
                            outputs = process_format_to_response(retrieval, answer)
                        except Exception as e:
                            outputs = error_format(
                                "내부 Excel 에 해당 자료가 없습니다.", 551
                            )
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
                            output = generate_answer(
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
                        except Exception as e:
                            outputs = error_format(
                                "내부 PPT에 해당 자료가 없습니다.", 552
                            )
                    future.set_result(outputs)
                except Exception as e:
                    future.set_result(error_format(f"처리 중 오류 발생: {str(e)}", 500))
