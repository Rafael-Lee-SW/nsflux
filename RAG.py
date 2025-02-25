# RAG.py
import torch
import re
import numpy as np
import rank_bm25
import random
import uuid
from datetime import datetime, timedelta
from sql import generate_sql

# Tracking
from tracking import time_tracker

# Import the vLLM to use the AsyncLLMEngine
from vllm.engine.async_llm_engine import AsyncLLMEngine

# In RAG.py (at the top, add an import for prompts)
from prompt_rag import QUERY_SORT_PROMPT, GENERATE_PROMPT_TEMPLATE, STREAM_PROMPT_TEMPLATE

global beep
beep = "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

@time_tracker
def execute_rag(QU, KE, TA, TI, **kwargs):
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    embed_model = kwargs.get("embed_model")
    embed_tokenizer = kwargs.get("embed_tokenizer")
    data = kwargs.get("data")
    config = kwargs.get("config")

    if TA == "yes":  # Table 이 필요하면
        # SQL
        final_sql_query, title, explain, table_json, chart_json = generate_sql(
            QU, model, tokenizer, config
        )

        # docs : 다음 LLM Input 으로 만들것 (String)
        PROMPT = f"""\
다음은 SQL 추출에 사용된 쿼리문이야 : {final_sql_query}. \
추가 설명 : {explain}. \
실제 SQL 추출된 데이터 : {str(table_json)}. \
"""
        # docs_list : 사용자들에게 보여줄 정보 (List)
        docs_list = [
            {"title": title, "data": table_json},
            {"title": "시각화 차트", "data": chart_json},
        ]

        return PROMPT, docs_list

    else:
        # RAG
        data = sort_by_time(TI, data)
        docs, docs_list = retrieve(KE, data, config.N, embed_model, embed_tokenizer)
        return docs, docs_list


@time_tracker
async def generate_answer(query, docs, **kwargs):
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    config = kwargs.get("config")
    
    answer = await generate(docs, query, model, tokenizer, config)
    return answer


@time_tracker
async def query_sort(params):
    # params: 딕셔너리로 전달된 값들
    query = params["user_input"]
    model = params["model"]
    tokenizer = params["tokenizer"]
    embed_model = params["embed_model"]
    embed_tokenizer = params["embed_tokenizer"]
    data = params["data"]
    config = params["config"]

    # prompts/prompt_rag.py에서 프롬프트 별도 관리
    PROMPT = QUERY_SORT_PROMPT.format(user_query=query)
    
    # Get Answer from LLM
    print("##### query_sort is starting #####")
    if config.use_vllm: # use_vllm = True case 
        from vllm import SamplingParams

        # 텍스트 생성 과정 제어 인자 설정 - 각 인자별 설명은 config.yaml 파일 참조
        sampling_params = SamplingParams(
            max_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
            top_k=config.model.top_k,
            top_p=config.model.top_p,
            repetition_penalty=config.model.repetition_penalty,
        )
        # request_id = 요청 주체를 구분하는 아이디, 유저별이 될 수도 있고, 대화별이 될 수도 있다.
        accepted_request_id = str(uuid.uuid4())

        # vllm을 통해 Model을 구동하여 Text를 생성한다.
        answer = await collect_vllm_text(PROMPT, model, sampling_params, accepted_request_id)
    else:
        # print(">>> About to Tokenizer")
        input_ids = tokenizer(
            PROMPT, return_tensors="pt", truncation=True, max_length=4024
        ).to("cuda")
        # print(">>> Finished tokenize")
        token_count = input_ids["input_ids"].shape[1]
        # print(f">>> Input token count: {token_count}", flush=True)
        # print(">>> About to call model.generate()")
        outputs = model.generate(
            **input_ids,
            max_new_tokens=config.model.max_new_tokens,
            do_sample=config.model.do_sample,
            temperature=config.model.temperature,
            top_k=config.model.top_k,
            top_p=config.model.top_p,
            repetition_penalty=config.model.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        # print(">>> Finished model.generate()")
        generated_tokens = outputs[0].shape[0]
        # print(f">>> Generated token count: {generated_tokens}")
        answer = tokenizer.decode(outputs[0][token_count:], skip_special_tokens=True)
        # print(answer)
        # print(">>> decode done, returning answer")

    # Regular expression to extract content between <query/> and <query>
    query_pattern = r"<query.*?>(.*?)<query.*?>"
    keyword_pattern = r"<keyword.*?>(.*?)<keyword.*?>"
    table_pattern = r"<table.*?>(.*?)<table.*?>"
    time_pattern = r"<time.*?>(.*?)<time.*?>"

    QU = re.search(query_pattern, answer, re.DOTALL).group(1)
    KE = re.search(keyword_pattern, answer, re.DOTALL).group(1)
    TA = re.search(table_pattern, answer, re.DOTALL).group(1)
    TI = re.search(time_pattern, answer, re.DOTALL).group(1)

    # 시간이 all 이면 전체에서 검색.
    if TI == "all":
        TI = "1900-01-01:2099-01-01"
    # print(beep)
    # print(f"사용자 질문:{query}")
    print(beep)
    print(f"구체화 질문: {QU}, 키워드 : {KE}, 테이블 필요 유무: {TA}, 시간: {TI}")
    print(beep)
    # Return the data as a tuple
    return QU, KE, TA, TI


@time_tracker
def sort_by_time(time_bound, data):
    date_format = "%Y-%m-%d"
    target_date_start = datetime.strptime(time_bound.split(":")[0], date_format)
    target_date_end = datetime.strptime(time_bound.split(":")[1], date_format)

    matching_indices = [
        i
        for i, date in enumerate(data["times"])
        if (not isinstance(date, str)) and (target_date_start < date < target_date_end)
    ]

    (
        data["file_names"],
        data["titles"],
        data["times"],
        data["vectors"],
        data["texts"],
        data["texts_short"],
        data["texts_vis"],
    ) = (
        [lst[i] for i in matching_indices]
        for lst in (
            data["file_names"],
            data["titles"],
            data["times"],
            data["vectors"],
            data["texts"],
            data["texts_short"],
            data["texts_vis"],
        )
    )
    return data


@time_tracker
def retrieve(query, data, N, embed_model, embed_tokenizer):
    # Similarity Score
    sim_score = cal_sim_score(query, data["vectors"], embed_model, embed_tokenizer)

    # BM25 Score
    bm25_score = cal_bm25_score(query, data["texts_short"], embed_tokenizer)

    # Scaling Scores
    scaled_sim_score = min_max_scaling(sim_score)
    scaled_bm25_score = min_max_scaling(bm25_score)

    # Total Score
    # score = (scaled_sim_score + scaled_bm25_score) / 2
    score = scaled_sim_score * 0.4 + scaled_bm25_score * 0.6
    top_k = score[:, 0, 0].argsort()[-N:][::-1]

    ## documents string 버전, dictionary 버전 둘 다 필요.
    documents = ""
    documents_list = []
    for i, index in enumerate(top_k):
        documents += f"{i+1}번째 검색자료 (출처:{data['file_names'][index]}) :\n{data['texts_short'][index]}\n"
        documents_list.append(
            {
                "file_name": data["file_names"][index],
                "title": data["titles"][index],
                "contents": data["texts_vis"][index],
            }
        )
        # print(
        #     f"{i+1}번째 검색자료 (출처:{data['file_names'][index]}) :\n{data['texts_short'][index]}"
        # )
        # print("\n" + beep)
    print("자료 검색 성공")
    return documents, documents_list


@time_tracker
def cal_sim_score(query, chunks, embed_model, embed_tokenizer):
    query_V = embed(query, embed_model, embed_tokenizer)
    if len(query_V.shape) == 1:
        query_V = query_V.unsqueeze(0)

    score = []
    for chunk in chunks:
        if len(chunk.shape) == 1:
            chunk = chunk.unsqueeze(0)

        query_norm = query_V / query_V.norm(dim=1)[:, None]
        chunk_norm = chunk / chunk.norm(dim=1)[:, None]
        tmp = torch.mm(query_norm, chunk_norm.transpose(0, 1)) * 100
        score.append(tmp.detach())

    return np.array(score)


@time_tracker
def cal_bm25_score(query, indexes, embed_tokenizer):
    tokenized_corpus = [
        embed_tokenizer(
            text,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_offsets_mapping=False,
        )
        for text in indexes
    ]
    tokenized_corpus = [
        embed_tokenizer.convert_ids_to_tokens(corpus["input_ids"])
        for corpus in tokenized_corpus
    ]

    bm25 = rank_bm25.BM25Okapi(tokenized_corpus)

    tokenized_query = embed_tokenizer(query)
    tokenized_query = embed_tokenizer.convert_ids_to_tokens(
        tokenized_query["input_ids"]
    )
    bm25_score = bm25.get_scores(tokenized_query)

    return np.array(bm25_score)


@time_tracker
def embed(query, embed_model, embed_tokenizer):
    inputs = embed_tokenizer(query, padding=True, truncation=True, return_tensors="pt")
    embeddings, _ = embed_model(**inputs, return_dict=False)
    return embeddings[0][0]


@time_tracker
def min_max_scaling(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


@time_tracker
async def generate(docs, query, model, tokenizer, config):
    
    # prompts/prompt_rag.py에서 프롬프트 별도 관리
    PROMPT = GENERATE_PROMPT_TEMPLATE.format(docs=docs, query=query)
    
    print("Inference steps")
    # vLLM 엔진 사용 시 (vLLM 엔진은 generate 메서드를 제공합니다)
    if config.use_vllm:
        from vllm import SamplingParams

        # 텍스트 생성 과정 제어 인자 설정 - 각 인자별 설명은 config.yaml 파일 참조
        sampling_params = SamplingParams(
            max_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
            top_k=config.model.top_k,
            top_p=config.model.top_p,
            repetition_penalty=config.model.repetition_penalty,
        )
        # request_id = 요청 주체를 구분하는 아이디, 유저별이 될 수도 있고, 대화별이 될 수도 있다.
        accepted_request_id = str(uuid.uuid4())
        
        # vllm을 통해 Model을 구동하여 Text를 생성한다.
        answer = await collect_vllm_text(PROMPT, model, sampling_params, accepted_request_id)
    else:
        # print(">>> About to Tokenizer")
        input_ids = tokenizer(
            PROMPT, return_tensors="pt", truncation=True, max_length=4024
        ).to("cuda")
        # print(">>> Finished tokenize")
        token_count = input_ids["input_ids"].shape[1]
        # print(f">>> Input token count: {token_count}", flush=True)
        # print(">>> About to call model.generate()")
        outputs = model.generate(
            **input_ids,
            max_new_tokens=config.model.max_new_tokens,
            do_sample=config.model.do_sample,
            temperature=config.model.temperature,
            top_k=config.model.top_k,
            top_p=config.model.top_p,
            repetition_penalty=config.model.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        # print(">>> Finished model.generate()")
        generated_tokens = outputs[0].shape[0]
        # print(f">>> Generated token count: {generated_tokens}")
        answer = tokenizer.decode(outputs[0][token_count:], skip_special_tokens=True)
        # print(answer)
        print(">>> decode done, returning answer")
    return answer


# vLLM을 통해 Text를 생성하고 모으는 함수
@time_tracker
async def collect_vllm_text(PROMPT, model, sampling_params, accepted_request_id):
    """
    Collects outputs from the vLLM model generation and extracts the final answer text.
    """
    import asyncio, concurrent.futures

    outputs = []  # List to store each generated output from the model.
    # Asynchronously iterate over generated outputs.
    async for output in model.generate(
        PROMPT, request_id=accepted_request_id, sampling_params=sampling_params
    ):
        outputs.append(output)  # 매 loop를 통해 생성되는 결과값을 저장한다. ---최적화 여지 있음---

    if not outputs:
        raise RuntimeError("No outputs were generated by the model.")

    # Finished=True일 때, vLLM의 Text 생성 작업이 종료되고, 해당 RequestOutput에 최종 Text가 담겨있다, 만약에 없을 경우 마지막 값이라도 담는다.
    final_output = next(
        (o for o in outputs if getattr(o, "finished", False)), outputs[-1]
    )
    # Extract the 'text' field from each CompletionOutput within the final RequestOutput.
    answer = "".join(
        [getattr(comp, "text", "") for comp in getattr(final_output, "outputs", [])]
    )
    # print("collect_vllm_text : Answer : ", answer)
    return answer


# -------------- NEW STREAMING VERSION --------------
@time_tracker
async def generate_answer_stream(query, docs, model, tokenizer, config):
    """
    Streaming version of generate_answer that yields partial tokens as they arrive.
    """
    # Build your prompt
    prompt = STREAM_PROMPT_TEMPLATE.format(docs=docs, query=query)

    if config.use_vllm:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
            top_k=config.model.top_k,
            top_p=config.model.top_p,
            repetition_penalty=config.model.repetition_penalty,
        )
        request_id = str(uuid.uuid4())

        # Instead of collecting the entire text, yield partial chunks:
        # vLLM engine usage:
        async for partial_chunk in collect_vllm_text_stream(prompt, model, sampling_params, request_id):
            yield partial_chunk

    else:
        # If not using vLLM, do HF generation with TextIteratorStreamer or similar:
        # Example code for HF:
        import torch
        from transformers import TextIteratorStreamer

        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4024
        ).to("cuda")
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        generation_kwargs = dict(
            **input_ids,
            streamer=streamer,
            max_new_tokens=config.model.max_new_tokens,
            do_sample=config.model.do_sample,
            temperature=config.model.temperature,
            top_k=config.model.top_k,
            top_p=config.model.top_p,
            repetition_penalty=config.model.repetition_penalty,
        )

        # Launch generation in background
        import threading
        t = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        t.start()
        # Now read partial tokens from streamer
        for new_token in streamer:
            yield new_token

@time_tracker
async def collect_vllm_text_stream(prompt, engine: AsyncLLMEngine, sampling_params, request_id) -> str:
    """
    vLLM-based streaming generator: yields each partial token or chunk
    as it arrives from the engine.
    """
    # We can watch each chunk in the async generator:
    async for request_output in engine.generate(prompt, request_id=request_id, sampling_params=sampling_params):
        # Each `request_output` can contain multiple completions, but usually you want the first one:
        # If you have multi-sampling or n=... then handle them all.
        if not request_output.outputs:
            continue
        # We can yield the partial text from the *last* token chunk
        for completion in request_output.outputs:
            yield completion.text  # or just the newly produced chunk
            
            # --------------- Streaming ------------------


if __name__ == "__main__":
    import asyncio

    # 엔진 생성 (예: engine_args는 미리 정의되어 있음)
    engine = AsyncLLMEngine.from_engine_args(engine_args, start_engine_loop=False)
    # 프로그램 시작 시 백그라운드 루프를 한 번만 시작합니다.
    if not engine.is_running:
        engine.start_background_loop()
    
    async def main():
        
        status = True
        while status == True:
            query = input("질문 : ")
            # QU, TA, TI = await query_sort(query)
            QU, TA, TI = await query_sort({"user_input": query, "model": None, "tokenizer": None, "embed_model": None, "embed_tokenizer": None, "data": None, "config": None})
            print("query_sort result done")
            if TA == "yes":  # Table 이 필요하면
                print("\n" + beep)
                SQL_results = generate_sql(QU)
                answer = await generate(SQL_results, query)
                print(answer)
                print("\n" + beep)
                print("\n" + beep)
                print("\n" + beep)

            else:
                file_names, titles, times, vectors, texts, texts_short = sort_by_time(
                    TI, file_names, titles, times, vectors, texts, texts_short
                )
                print("\n" + beep)
                docs = retrieve(QU, vectors, texts, texts_short, file_names, N)
                print("\n" + beep)
                answer = await generate(docs, query)
                print(answer)
                print("\n" + beep)
                print("\n" + beep)
                print("\n" + beep)
    asyncio.run(main())    