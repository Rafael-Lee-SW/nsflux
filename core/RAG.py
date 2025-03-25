# core/RAG.py
import torch
import re
import numpy as np
import rank_bm25
import random
import uuid
import logging
from datetime import datetime, timedelta
# from sql import generate_sql  # (구) 제거된 import

# Tracking
from utils.tracking import time_tracker

# Import the vLLM to use the AsyncLLMEngine
from vllm.engine.async_llm_engine import AsyncLLMEngine
# 이미지 임베딩을 별도로 계산하여 프롬프트에 포함하기
from vllm.model_executor.models.interfaces import SupportsMultiModal

# Prompt 템플릿 불러오기
from prompt import QUERY_SORT_PROMPT, GENERATE_PROMPT_TEMPLATE, STREAM_PROMPT_TEMPLATE, SQL_EXTRACTION_PROMPT_TEMPLATE
# SQL만 담당하는 함수들만 import
from core.SQL_NS import run_sql_unno, run_sql_bl, get_metadata  

global beep
beep = "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

@time_tracker
async def execute_rag(QU, KE, TA, TI, **kwargs):
    print("[SOOWAN]: execute_rag : 진입")
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    embed_model = kwargs.get("embed_model")
    embed_tokenizer = kwargs.get("embed_tokenizer")
    data = kwargs.get("data")
    config = kwargs.get("config")
    
    if TA == "yes":  # Table 이 필요하면
        print("[SOOWAN]: execute_rag : 테이블 필요 (TA == yes). SQL 생성 시작합니다.")
        try:
            # generate_sql 함수를 이 파일(RAG.py) 내부에 새로 정의했으므로, 여기서 직접 호출
            result = await generate_sql(QU, model, tokenizer, config)
        except Exception as e:
            # 1) generate_sql() 자체가 도중에 예외를 던지는 경우
            print("[ERROR] generate_sql() 도중 예외 발생:", e)
            # 멈추지 않고, 에러 형식으로 데이터를 만들어 반환
            docs = (
                "테이블 조회 시도 중 예외가 발생했습니다. "
                "해당 SQL을 실행할 수 없어서 테이블 데이터를 가져오지 못했습니다."
            )
            docs_list = []
            return docs, docs_list

        # 2) 함수가 정상 실행됐지만 결과가 None인 경우(= SQL 쿼리 결과가 없거나 오류)
        if result is None:
            print(
                "[WARNING] generate_sql()에서 None을 반환했습니다. "
                "SQL 수행 결과가 없거나 에러가 발생한 것일 수 있습니다."
            )
            docs = (
                "테이블 조회 결과가 비어 있습니다. "
                "조회할 데이터가 없거나 SQL 오류가 발생했습니다."
            )
            docs_list = []
            return docs, docs_list

        # 기존 generate_sql은 이제 6개의 값을 반환합니다.
        final_sql_query, title, explain, table_json, chart_json, detailed_result = result

        # docs : LLM 입력용 (string)
        PROMPT = (
            f"실제 사용된 SQL문: {final_sql_query}\n\n"
            f"추가 설명: {explain}\n\n"
            f"실제 SQL 추출된 데이터: {str(table_json)}\n\n"
            f"실제 선적된 B/L 데이터: {str(detailed_result)}\n\n"
        )
        # docs_list에 DG B/L 상세 정보를 추가하여 총 3개 항목으로 구성합니다.
        docs_list = [
            {"title": title, "data": table_json},
            {"title": "DG B/L 상세 정보", "data": detailed_result},
        ]
        print("[SOOWAN]: execute_rag : 테이블 부분 정상 처리 완료")

        return PROMPT, docs_list

    else:
        print("[SOOWAN]: execute_rag : 테이블 필요없음")
        # 적응형 시간 필터링으로 RAG 실행
        filtered_data = expand_time_range_if_needed(TI, data, min_docs=50)

        # 디버깅을 위해 문서 수 로깅
        print(f"[RETRIEVE] 검색에 사용되는 문서 수: {len(filtered_data.get('vectors', []))}")

        docs, docs_list = retrieve(KE, filtered_data, config.N, embed_model, embed_tokenizer)
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
    max_attempts = 3
    attempt = 0
    while attempt < max_attempts:
        # params: 딕셔너리로 전달된 값들
        query = params["user_input"]
        model = params["model"]
        tokenizer = params["tokenizer"]
        embed_model = params["embed_model"]
        embed_tokenizer = params["embed_tokenizer"]
        data = params["data"]
        config = params["config"]

        # 프롬프트 생성
        PROMPT = QUERY_SORT_PROMPT.format(user_query=query)
        print("##### query_sort is starting, attempt:", attempt + 1, "#####")

        # Get Answer from LLM
        if config.use_vllm:  # use_vllm = True case
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                max_tokens=config.model.max_new_tokens,
                temperature=config.model.temperature,
                top_k=config.model.top_k,
                top_p=config.model.top_p,
                repetition_penalty=config.model.repetition_penalty,
            )
            accepted_request_id = str(uuid.uuid4())
            answer = await collect_vllm_text(PROMPT, model, sampling_params, accepted_request_id)
        else:
            input_ids = tokenizer(
                PROMPT, return_tensors="pt", truncation=True, max_length=4024
            ).to("cuda")
            token_count = input_ids["input_ids"].shape[1]
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
            answer = tokenizer.decode(outputs[0][token_count:], skip_special_tokens=True)

        print("[DEBUG query_sort] Generated answer:")
        print(answer)

        # Regular expressions for tags
        query_pattern = r"<query.*?>(.*?)<query.*?>"
        keyword_pattern = r"<keyword.*?>(.*?)<keyword.*?>"
        table_pattern = r"<table.*?>(.*?)<table.*?>"
        time_pattern = r"<time.*?>(.*?)<time.*?>"

        m_query = re.search(query_pattern, answer, re.DOTALL)
        m_keyword = re.search(keyword_pattern, answer, re.DOTALL)
        m_table = re.search(table_pattern, answer, re.DOTALL)
        m_time = re.search(time_pattern, answer, re.DOTALL)

        if m_query and m_keyword and m_table and m_time:
            QU = m_query.group(1).strip()
            KE = m_keyword.group(1).strip()
            TA = m_table.group(1).strip()
            TI = m_time.group(1).strip()
            if TI == "all":
                TI = "1900-01-01:2099-01-01"
            print(beep)
            print(f"구체화 질문: {QU}, 키워드 : {KE}, 테이블 필요 유무: {TA}, 시간: {TI}")
            print(beep)
            return QU, KE, TA, TI
        else:
            print("[ERROR query_sort] 필요한 태그들이 누락되었습니다. 재시도합니다.")
            attempt += 1

    # 3회 재시도 후에도 실패하면 에러 발생
    raise ValueError("LLM이 올바른 태그 형식의 답변을 생성하지 못했습니다.")


@time_tracker
async def specific_question(params):
    """
    query_sort와 동일한 로직을 수행할 수도 있으나,
    별도로 분리된 이유가 있다면 여기서 추가 처리 가능
    """
    max_attempts = 3
    attempt = 0
    while attempt < max_attempts:
        query = params["user_input"]
        model = params["model"]
        tokenizer = params["tokenizer"]
        embed_model = params["embed_model"]
        embed_tokenizer = params["embed_tokenizer"]
        data = params["data"]
        config = params["config"]

        PROMPT = QUERY_SORT_PROMPT.format(user_query=query)
        print("##### query_sort is starting, attempt:", attempt + 1, "#####")

        if config.use_vllm:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                max_tokens=config.model.max_new_tokens,
                temperature=config.model.temperature,
                top_k=config.model.top_k,
                top_p=config.model.top_p,
                repetition_penalty=config.model.repetition_penalty,
            )
            accepted_request_id = str(uuid.uuid4())
            answer = await collect_vllm_text(PROMPT, model, sampling_params, accepted_request_id)
        else:
            input_ids = tokenizer(
                PROMPT, return_tensors="pt", truncation=True, max_length=4024
            ).to("cuda")
            token_count = input_ids["input_ids"].shape[1]
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
            answer = tokenizer.decode(outputs[0][token_count:], skip_special_tokens=True)

        print("[DEBUG query_sort] Generated answer:")
        print(answer)

        query_pattern = r"<query.*?>(.*?)<query.*?>"
        keyword_pattern = r"<keyword.*?>(.*?)<keyword.*?>"
        table_pattern = r"<table.*?>(.*?)<table.*?>"
        time_pattern = r"<time.*?>(.*?)<time.*?>"

        m_query = re.search(query_pattern, answer, re.DOTALL)
        m_keyword = re.search(keyword_pattern, answer, re.DOTALL)
        m_table = re.search(table_pattern, answer, re.DOTALL)
        m_time = re.search(time_pattern, answer, re.DOTALL)

        if m_query and m_keyword and m_table and m_time:
            QU = m_query.group(1).strip()
            KE = m_keyword.group(1).strip()
            TA = m_table.group(1).strip()
            TI = m_time.group(1).strip()
            if TI == "all":
                TI = "1900-01-01:2099-01-01"
            print(beep)
            print(f"구체화 질문: {QU}, 키워드 : {KE}, 테이블 필요 유무: {TA}, 시간: {TI}")
            print(beep)
            return QU, KE, TA, TI
        else:
            print("[ERROR query_sort] 필요한 태그들이 누락되었습니다. 재시도합니다.")
            attempt += 1

    raise ValueError("LLM이 올바른 태그 형식의 답변을 생성하지 못했습니다.")


@time_tracker
def sort_by_time(time_bound, data):
    """
    원본 데이터는 유지하고 필터링된 복사본을 반환하는 함수
    """
    original_count = len(data["times"])
    print(f"[시간 필터 전] 문서 수: {original_count}")

    if time_bound == "all" or time_bound == "1900-01-01:2099-01-01":
        print(f"[시간 필터] 전체 기간 사용 - 모든 문서 포함")
        return data  # 원본 그대로

    date_format = "%Y-%m-%d"
    target_date_start = datetime.strptime(time_bound.split(":")[0], date_format)
    target_date_end = datetime.strptime(time_bound.split(":")[1], date_format)

    matching_indices = [
        i
        for i, date in enumerate(data["times"])
        if (not isinstance(date, str)) and (target_date_start < date < target_date_end)
    ]

    filtered_count = len(matching_indices)
    print(f"[시간 필터 후] 문서 수: {filtered_count}, 기간: {time_bound}")

    if filtered_count < 50 and filtered_count < original_count * 0.1:
        print(f"[경고] 시간 필터로 인해 문서가 크게 줄었습니다: {original_count} → {filtered_count}")

    filtered_data = {}
    filtered_data["file_names"] = [data["file_names"][i] for i in matching_indices]
    filtered_data["titles"] = [data["titles"][i] for i in matching_indices]
    filtered_data["times"] = [data["times"][i] for i in matching_indices]
    filtered_data["chunk_ids"] = [data["chunk_ids"][i] for i in matching_indices]

    if isinstance(data["vectors"], torch.Tensor):
        filtered_data["vectors"] = data["vectors"][matching_indices]
    else:
        filtered_data["vectors"] = [data["vectors"][i] for i in matching_indices]

    filtered_data["texts"] = [data["texts"][i] for i in matching_indices]
    filtered_data["texts_short"] = [data["texts_short"][i] for i in matching_indices]
    filtered_data["texts_vis"] = [data["texts_vis"][i] for i in matching_indices]

    return filtered_data


@time_tracker
def retrieve(query, data, N, embed_model, embed_tokenizer):
    print("[SOOWAN] retrieve : 진입")
    logging.info(f"Retrieval for query: '{query}'")
    logging.info(f"Available documents: {len(data['vectors'])}")

    try:
        sim_score = cal_sim_score(query, data["vectors"], embed_model, embed_tokenizer)
        logging.info(f"Similarity score shape: {sim_score.shape}")

        bm25_score = cal_bm25_score(query, data["texts_short"], embed_tokenizer)
        logging.info(f"BM25 score shape: {bm25_score.shape}")

        scaled_sim_score = min_max_scaling(sim_score)
        scaled_bm25_score = min_max_scaling(bm25_score)

        score = scaled_sim_score * 0.4 + scaled_bm25_score * 0.6
        score_values = score[:, 0, 0]
        top_k = score[:, 0, 0].argsort()[-N:][::-1]

        logging.info(f"Top {N} document indices: {top_k}")
        logging.info(f"Top {N} document scores: {[score[:, 0, 0][i] for i in top_k]}")
        logging.info(f"Top document titles: {[data['titles'][i] for i in top_k]}")

        documents = ""
        documents_list = []
        for i, index in enumerate(top_k):
            score_str = f"{score_values[index]:.4f}"
            documents += f"{i+1}번째 검색자료 (출처:{data['file_names'][index]}) :\n{data['texts_short'][index]}, , Score: {score_str}\n"
            documents_list.append(
                {
                    "file_name": data["file_names"][index],
                    "title": data["titles"][index],
                    "contents": data["texts_vis"][index],
                    "chunk_id": data["chunk_ids"][index],
                }
            )
        print("-------------자료 검색 성공--------------")
        print("-------", documents_list, "-------")
        print("---------------------------------------")
        return documents, documents_list

    except Exception as e:
        logging.error(f"Retrieval error: {str(e)}", exc_info=True)
        return "", []


@time_tracker
def expand_time_range_if_needed(time_bound, data, min_docs=50):
    """
    시간 필터링 결과가 너무 적은 경우 자동으로 시간 범위를 확장하는 함수
    """
    if time_bound == "all" or time_bound == "1900-01-01:2099-01-01":
        print(f"[시간 범위] 전체 기간 사용")
        return data

    filtered_data = sort_by_time(time_bound, data)
    filtered_count = len(filtered_data.get("times", []))

    if filtered_count >= min_docs:
        print(f"[시간 범위] 원래 범위로 충분한 문서 확보: {filtered_count}개")
        return filtered_data

    print(f"[시간 범위 확장] 원래 범위는 {filtered_count}개 문서만 제공 (최소 필요: {min_docs}개)")

    date_format = "%Y-%m-%d"
    try:
        start_date = datetime.strptime(time_bound.split(":")[0], date_format)
        end_date = datetime.strptime(time_bound.split(":")[1], date_format)
    except Exception as e:
        print(f"[시간 범위 오류] 날짜 형식 오류: {time_bound}, 오류: {e}")
        return data

    expansions = [
        (3, "3개월"),
        (6, "6개월"),
        (12, "1년"),
        (24, "2년"),
        (60, "5년"),
    ]

    for months, label in expansions:
        new_start = start_date - timedelta(days=30 * months // 2)
        new_end = end_date + timedelta(days=30 * months // 2)

        new_range = f"{new_start.strftime(date_format)}:{new_end.strftime(date_format)}"
        print(f"[시간 범위 확장] {label} 확장 시도: {new_range}")

        expanded_data = sort_by_time(new_range, data)
        expanded_count = len(expanded_data.get("times", []))

        if expanded_count >= min_docs:
            print(f"[시간 범위 확장] {label} 확장으로 {expanded_count}개 문서 확보")
            return expanded_data

    print(f"[시간 범위 확장] 모든 확장 시도 실패, 전체 데이터셋 사용")
    return data


@time_tracker
def cal_sim_score(query, chunks, embed_model, embed_tokenizer):
    print("[SOOWAN] cal_sim_score : 진입 / query : ", query)
    query_V = embed(query, embed_model, embed_tokenizer)
    print("[SOOWAN] cal_sim_score : query_V 생산 완료")
    if len(query_V.shape) == 1:
        query_V = query_V.unsqueeze(0)
        print("[SOOWAN] cal_sim_score : query_V.shape == 1")
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
    logging.info(f"Starting BM25 calculation for query: {query}")
    logging.info(f"Document count: {len(indexes)}")

    if not indexes:
        logging.warning("Empty document list provided to BM25")
        return np.zeros(0)

    tokenized_corpus = []
    for i, text in enumerate(indexes):
        try:
            tokens = embed_tokenizer(
                text,
                return_token_type_ids=False,
                return_attention_mask=False,
                return_offsets_mapping=False,
            )
            tokens = embed_tokenizer.convert_ids_to_tokens(tokens["input_ids"])
            if len(tokens) == 0:
                logging.warning(f"Document {i} tokenized to empty list")
                tokens = ["<empty>"]
            tokenized_corpus.append(tokens)
        except Exception as e:
            logging.error(f"Failed to tokenize document {i}: {str(e)}")
            tokenized_corpus.append(["<error>"])

    try:
        bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
        tokenized_query = embed_tokenizer.convert_ids_to_tokens(embed_tokenizer(query)["input_ids"])
        scores = bm25.get_scores(tokenized_query)

        if np.isnan(scores).any() or np.isinf(scores).any():
            logging.warning("BM25 produced NaN/Inf scores - replacing with zeros")
            scores = np.nan_to_num(scores)

        logging.info(
            f"BM25 scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}"
        )
        return scores
    except Exception as e:
        logging.error(f"BM25 scoring failed: {str(e)}")
        return np.zeros(len(indexes))


@time_tracker
def embed(query, embed_model, embed_tokenizer):
    print("[SOOWAN] embed: 진입")
    inputs = embed_tokenizer(query, padding=True, truncation=True, return_tensors="pt")
    embeddings, _ = embed_model(**inputs, return_dict=False)
    print("[SOOWAN] embed: 완료")
    return embeddings[0][0]


@time_tracker
def min_max_scaling(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max == arr_min:
        print("[SOOWAN] min_max_scaling: Zero range detected, returning zeros.")
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


@time_tracker
async def generate(docs, query, model, tokenizer, config):
    PROMPT = GENERATE_PROMPT_TEMPLATE.format(docs=docs, query=query)
    print("Inference steps")
    if config.use_vllm:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
            top_k=config.model.top_k,
            top_p=config.model.top_p,
            repetition_penalty=config.model.repetition_penalty,
        )
        accepted_request_id = str(uuid.uuid4())
        answer = await collect_vllm_text(PROMPT, model, sampling_params, accepted_request_id)
    else:
        input_ids = tokenizer(PROMPT, return_tensors="pt", truncation=True, max_length=4024).to(
            "cuda"
        )
        token_count = input_ids["input_ids"].shape[1]
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
        answer = tokenizer.decode(outputs[0][token_count:], skip_special_tokens=True)
        print(answer)
        print(">>> decode done, returning answer")
    return answer


@time_tracker
async def collect_vllm_text(PROMPT, model, sampling_params, accepted_request_id):
    import asyncio, concurrent.futures
    print("[SOOWAN] collect_vllm_text 진입 PROMPT: ")
    outputs = []
    async for output in model.generate(PROMPT, request_id=accepted_request_id, sampling_params=sampling_params):
        outputs.append(output)
    if not outputs:
        raise RuntimeError("No outputs were generated by the model.")
    final_output = next((o for o in outputs if getattr(o, "finished", False)), outputs[-1])
    answer = "".join([getattr(comp, "text", "") for comp in getattr(final_output, "outputs", [])])
    return answer


@time_tracker
async def generate_answer_stream(query, docs, model, tokenizer, config):
    
    prompt = STREAM_PROMPT_TEMPLATE.format(docs=docs, query=query)
    print("최종 LLM 추론용 prompt 생성 : ", prompt)
    
    import uuid
    import base64
    import io
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import SamplingParams
    from vllm.multimodal.utils import fetch_image
    
    print("[IMAGE-STREAMING-QUERY] Image_query 진입")
    
    image_input = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 2) Image -> PIL
    pil_image = None
    try:
        print("[DEBUG] Step2 - Converting image data to PIL...")
        if isinstance(image_input, str) and (
            image_input.startswith("http://") or image_input.startswith("https://")
        ):
            print("[DEBUG]   => image_input is a URL; using fetch_image()")
            pil_image = fetch_image(image_input)
        else:
            print("[DEBUG]   => image_input is presumably base64")
            if isinstance(image_input, str) and image_input.startswith("data:image/"):
                print("[DEBUG]   => detected 'data:image/' prefix => splitting off base64 header")
                image_input = image_input.split(",", 1)[-1]
            decoded = base64.b64decode(image_input)
            print(f"[DEBUG]   => decoded base64 length={len(decoded)} bytes")
            pil_image = Image.open(io.BytesIO(decoded)).convert("RGB")
        print("[DEBUG] Step2 - PIL image loaded successfully:", pil_image.size)
    except Exception as e:
        err_msg = f"[ERROR-step2] Failed to load image: {str(e)}"
        print(err_msg)
    
    # 3) HF Processor 로드
    try:
        print(f"[DEBUG] Step3 - Loading processor from '{config.model_id}' ... (use_fast=False)")
        processor = AutoProcessor.from_pretrained(
            config.model_id,
            use_fast=False  # 모델이 fast processor를 지원한다면 True로 시도 가능
        )
        print("[DEBUG]   => processor loaded:", type(processor).__name__)
    except Exception as e:
        err_msg = f"[ERROR-step3] Failed to load processor: {str(e)}"
        print(err_msg)
    
    # 4) Chat Template 적용 (tokenize=False => 최종 prompt string만 얻음)
    print("[DEBUG] Step4 - Constructing messages & applying chat template (tokenize=False)...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": pil_image},   # PIL object
                {"type": "text",  "text": prompt}, # 사용자 질의
            ],
        }
    ]
    
    try:
        # tokenize=False => prompt를 'raw string' 형태로 얻음
        prompt_string = processor.apply_chat_template(
            messages,
            tokenize=False,           # 핵심!
            add_generation_prompt=True,
        )
        print("[DEBUG]   => prompt_string (first ~200 chars):", prompt_string[:200])
    except Exception as e:
        err_msg = f"[ERROR-step4] Error in processor.apply_chat_template: {str(e)}"
        print(err_msg)
    
    generate_request = {
        "prompt": prompt_string,
        "multi_modal_data": {
            "image": [pil_image]  # 여러 장이라면 list에 더 추가
        }
    }
    
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
        async for partial_chunk in collect_vllm_text_stream(generate_request, model, sampling_params, request_id):
            yield partial_chunk
    else:
        import torch
        from transformers import TextIteratorStreamer

        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4024).to("cuda")
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
        import threading

        t = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        t.start()
        for new_token in streamer:
            yield new_token

@time_tracker
async def collect_vllm_text_stream(prompt, engine: AsyncLLMEngine, sampling_params, request_id) -> str:
    async for request_output in engine.generate(prompt, request_id=request_id, sampling_params=sampling_params):
        if not request_output.outputs:
            continue
        for completion in request_output.outputs:
            yield completion.text

# -------------------------------------------------------------------------
# PROCESS IMAGE QUERY (IMAGE TO TEXT)
# -------------------------------------------------------------------------
@time_tracker
async def image_query(http_query, model, tokenizer, config):
    """
    기존 ray_utils.py의 process_image_query 로직을 옮겨온 함수.
    이미지 입력을 받아 최종 한 번에 답변을 생성(스트리밍 없음).
    """
    import uuid
    import base64
    import io
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import SamplingParams
    from vllm.multimodal.utils import fetch_image
    
    print("[IMAGE-NONSTREAMING-QUERY] Image_query 진입")
    
    # 1) 파라미터 파싱
    request_id = http_query.get("request_id", str(uuid.uuid4()))
    image_input = http_query.get("image_data")
    user_query = http_query.get("qry_contents", "이 이미지를 한국어로 잘 설명해주세요.")
    print(f"[DEBUG] Step1 - user_query='{user_query}', image_input type={type(image_input)}")
    
    # 2) Image -> PIL
    pil_image = None
    try:
        print("[DEBUG] Step2 - Converting image data to PIL...")
        if isinstance(image_input, str) and (
            image_input.startswith("http://") or image_input.startswith("https://")
        ):
            print("[DEBUG]   => image_input is a URL; using fetch_image()")
            pil_image = fetch_image(image_input)
        else:
            print("[DEBUG]   => image_input is presumably base64")
            if isinstance(image_input, str) and image_input.startswith("data:image/"):
                print("[DEBUG]   => detected 'data:image/' prefix => splitting off base64 header")
                image_input = image_input.split(",", 1)[-1]
            decoded = base64.b64decode(image_input)
            print(f"[DEBUG]   => decoded base64 length={len(decoded)} bytes")
            pil_image = Image.open(io.BytesIO(decoded)).convert("RGB")
        print("[DEBUG] Step2 - PIL image loaded successfully:", pil_image.size)
    except Exception as e:
        err_msg = f"[ERROR-step2] Failed to load image: {str(e)}"
        print(err_msg)
        return {"type": "error", "message": err_msg}
    
    # 3) HF Processor 로드
    try:
        print(f"[DEBUG] Step3 - Loading processor from '{config.model_id}' ... (use_fast=False)")
        processor = AutoProcessor.from_pretrained(
            config.model_id,
            use_fast=False  # 모델이 fast processor를 지원한다면 True로 시도 가능
        )
        print("[DEBUG]   => processor loaded:", type(processor).__name__)
    except Exception as e:
        err_msg = f"[ERROR-step3] Failed to load processor: {str(e)}"
        print(err_msg)
        return {"type": "error", "message": err_msg}
    
    # 4) Chat Template 적용 (tokenize=False => 최종 prompt string만 얻음)
    print("[DEBUG] Step4 - Constructing messages & applying chat template (tokenize=False)...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": pil_image},   # PIL object
                {"type": "text",  "text": user_query}, # 사용자 질의
            ],
        }
    ]
    
    try:
        # tokenize=False => prompt를 'raw string' 형태로 얻음
        prompt_string = processor.apply_chat_template(
            messages,
            tokenize=False,           # 핵심!
            add_generation_prompt=True,
        )
        print("[DEBUG]   => prompt_string (first ~200 chars):", prompt_string[:200])
    except Exception as e:
        err_msg = f"[ERROR-step4] Error in processor.apply_chat_template: {str(e)}"
        print(err_msg)
        return {"type": "error", "message": err_msg}
    
    # 5) Sampling Params
    print("[DEBUG] Step5 - Setting sampling params...")
    sampling_params = SamplingParams(
        max_tokens=config.model.max_new_tokens,
        temperature=config.model.temperature,
        top_k=config.model.top_k,
        top_p=config.model.top_p,
        repetition_penalty=config.model.repetition_penalty,
    )
    print("[DEBUG]   => sampling_params =", sampling_params)
    
    # 6) Generate 호출
    print("[DEBUG] Step6 - Starting vLLM generate(...) using multi_modal_data")
    result_chunks = []
    try:
        # vLLM에 prompt와 함께 multi_modal_data를 넘김
        # => vLLM 내부에서 Gemma3MultiModalProcessor가 image 임베딩 및 token placement 처리
        generate_request = {
            "prompt": prompt_string,
            "multi_modal_data": {
                "image": [pil_image]  # 여러 장이라면 list에 더 추가
            }
        }

        async for out in model.generate(
            prompt=generate_request,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            result_chunks.append(out)
        print("[DEBUG] Step6 - All chunks retrieved: total:", len(result_chunks))

    except Exception as e:
        err_msg = f"[ERROR-step6] Error in model.generate: {str(e)}"
        print(err_msg)
        return {"type": "error", "message": err_msg}

    if not result_chunks:
        print("[DEBUG] Step6 - No output from model => returning error")
        return {"type": "error", "message": "No output from model."}

    final_output = next((c for c in result_chunks if getattr(c, "finished", False)), result_chunks[-1])
    answer_text = "".join(piece.text for piece in final_output.outputs)
    print(f"[DEBUG] Final answer: {answer_text}")

    # 완료
    print(f"[DEBUG] [process_image_query] DONE => request_id={request_id}")
    return {
        "result": answer_text,
        "request_id": request_id,
        "status_code": 200
    }

@time_tracker
async def image_streaming_query(http_query, model, tokenizer, config):
    """
    새로 추가된 이미지 스트리밍 함수:
    - 이미지 입력 + 사용자 텍스트를 받아서, 부분 토큰을 SSE로 전달할 수 있도록 yield 함.
    - use_vllm=True일 때는 collect_vllm_text_stream와 동일한 방식으로 partial chunk를 yield
    - HF standard는 TextIteratorStreamer를 이용한 방식으로 partial chunk를 yield
    """
    import uuid
    import base64
    import io
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import SamplingParams
    from vllm.multimodal.utils import fetch_image

    print("[IMAGE-STREAMING-QUERY] Image_streaming_query 진입")

    # 1) 파라미터 파싱
    request_id = http_query.get("request_id", str(uuid.uuid4()))
    image_input = http_query.get("image_data")
    user_query = http_query.get("qry_contents", "이 이미지를 설명해주세요.")
    print(f"[DEBUG] Step1 - user_query='{user_query}', image_input type={type(image_input)}")
    
    # 2) Image -> PIL
    try:
        print("[DEBUG] Step2 - Converting image data to PIL...")
        if isinstance(image_input, str) and (image_input.startswith("http://") or image_input.startswith("https://")):
            print("[DEBUG]   => image_input is a URL; using fetch_image()")
            pil_image = fetch_image(image_input)
        else:
            print("[DEBUG]   => image_input is presumably base64")
            if isinstance(image_input, str) and image_input.startswith("data:image/"):
                print("[DEBUG]   => detected 'data:image/' prefix => splitting off base64 header")
                image_input = image_input.split(",", 1)[-1]
            decoded = base64.b64decode(image_input)
            print(f"[DEBUG]   => decoded base64 length={len(decoded)} bytes")
            pil_image = Image.open(io.BytesIO(decoded)).convert("RGB")
        print("[DEBUG] Step2 - PIL image loaded successfully:", pil_image.size)
    except Exception as e:
        err_msg = f"[ERROR-step2] Failed to load image: {str(e)}"
        print(err_msg)
        yield {"type": "error", "message": err_msg}
        return  # No value, simply return

    # 3) HF Processor 로드
    try:
        print(f"[DEBUG] Step3 - Loading processor from '{config.model_id}' ... (use_fast=False)")
        processor = AutoProcessor.from_pretrained(config.model_id, use_fast=False)
        print("[DEBUG]   => processor loaded:", type(processor).__name__)
    except Exception as e:
        err_msg = f"[ERROR-step3] Failed to load processor: {str(e)}"
        print(err_msg)
        yield {"type": "error", "message": err_msg}
        return

    # 4) Chat Template 적용 (tokenize=False => 최종 prompt string만 얻음)
    print("[DEBUG] Step4 - Constructing messages & applying chat template (tokenize=False)...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": pil_image},
                {"type": "text",  "text": user_query},
            ],
        }
    ]
    try:
        prompt_string = processor.apply_chat_template(
            messages,
            tokenize=False,           # 핵심!
            add_generation_prompt=True,
        )
        print("[DEBUG]   => prompt_string (first ~200 chars):", prompt_string[:200])
    except Exception as e:
        err_msg = f"[ERROR-step4] Error in processor.apply_chat_template: {str(e)}"
        print(err_msg)
        yield {"type": "error", "message": err_msg}
        return

    # 5) Sampling Params
    print("[DEBUG] Step5 - Setting sampling params...")
    sampling_params = SamplingParams(
        max_tokens=config.model.max_new_tokens,
        temperature=config.model.temperature,
        top_k=config.model.top_k,
        top_p=config.model.top_p,
        repetition_penalty=config.model.repetition_penalty,
    )
    print("[DEBUG]   => sampling_params =", sampling_params)
    
    # 6) Generate 호출
    print("[DEBUG] Step6 - Starting vLLM generate(...) using multi_modal_data")
    generate_request = {
        "prompt": prompt_string,
        "multi_modal_data": {
            "image": [pil_image]
        }
    }
    async for partial_chunk in collect_vllm_text_stream(generate_request, model, sampling_params, request_id):
        yield partial_chunk

# ---------------------------
# *** 새로 추가된 generate_sql 함수 ***
# *** (기존에는 SQL_NS.py 안에 있었음) ***
# ---------------------------
@time_tracker
async def generate_sql(user_query, model, tokenizer, config):
    """
    기존 SQL_NS.py에서 VLLM을 통해 <unno>, <class>, <pol_port>, <pod_port>를 추출한 뒤
    run_sql_unno, run_sql_bl를 호출하는 로직을 RAG.py로 옮겼습니다.
    """
    # 먼저 메타데이터를 읽어옴
    metadata_location, metadata_unno = get_metadata(config)
    print("")
    # 프롬프트 생성
    PROMPT = SQL_EXTRACTION_PROMPT_TEMPLATE.format(metadata_location=metadata_location, query=user_query)

    from vllm import SamplingParams
    import uuid

    sampling_params = SamplingParams(
        max_tokens=config.model.max_new_tokens,
        temperature=config.model.temperature,
        top_k=config.model.top_k,
        top_p=config.model.top_p,
        repetition_penalty=config.model.repetition_penalty,
    )

    # 최대 3회 시도
    max_attempts = 3
    attempt = 0
    UN_number = UN_class = POL = POD = "NULL"
    unno_pattern = r'<unno.*?>(.*?)<unno.*?>'
    class_pattern = r'<class.*?>(.*?)<class.*?>'
    pol_port_pattern = r'<pol_port.*?>(.*?)<pol_port.*?>'
    pod_port_pattern = r'<pod_port.*?>(.*?)<pod_port.*?>'

    while attempt < max_attempts:
        accepted_request_id = str(uuid.uuid4())
        outputs_result = await collect_vllm_text(PROMPT, model, sampling_params, accepted_request_id)
        print(f"[GENERATE_SQL] Attempt {attempt+1}, SQL Model Outputs: {outputs_result}")

        match_unno = re.search(unno_pattern, outputs_result, re.DOTALL)
        UN_number = match_unno.group(1).strip() if match_unno else "NULL"

        match_class = re.search(class_pattern, outputs_result, re.DOTALL)
        UN_class = match_class.group(1).strip() if match_class else "NULL"

        match_pol = re.search(pol_port_pattern, outputs_result, re.DOTALL)
        POL = match_pol.group(1).strip() if match_pol else "NULL"

        match_pod = re.search(pod_port_pattern, outputs_result, re.DOTALL)
        POD = match_pod.group(1).strip() if match_pod else "NULL"

        print(f"[GENERATE_SQL] 추출 결과 - UN_number: {UN_number}, UN_class: {UN_class}, POL: {POL}, POD: {POD}")

        # 조건: (UN_number != NULL or UN_class != NULL) and (POL != NULL) and (POD != NULL)
        if ((UN_number != "NULL" or UN_class != "NULL") and POL != "NULL" and POD != "NULL"):
            break
        attempt += 1

    print(f"[GENERATE_SQL] 최종 추출 값 - UN_number: {UN_number}, UN_class: {UN_class}, POL: {POL}, POD: {POD}")

    # run_sql_unno로 DG 가능 여부 확인
    final_sql_query, result = run_sql_unno(UN_class, UN_number, POL, POD)
    # 상세 B/L SQL
    detailed_sql_query, detailed_result = run_sql_bl(UN_class, UN_number, POL, POD)

    # Temporary: title, explain, table_json, chart_json = None
    title, explain, table_json, chart_json = (None,) * 4

    return final_sql_query, title, explain, result, chart_json, detailed_result
