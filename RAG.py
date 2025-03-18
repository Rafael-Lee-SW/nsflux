# RAG.py
import torch
import re
import numpy as np
import rank_bm25
import random
import uuid
import logging
from datetime import datetime, timedelta
# from sql import generate_sql
from SQL_NS import generate_sql

# Tracking
from tracking import time_tracker

# Import the vLLM to use the AsyncLLMEngine
from vllm.engine.async_llm_engine import AsyncLLMEngine

# In RAG.py (at the top, add an import for prompts)
from prompt_rag import QUERY_SORT_PROMPT, GENERATE_PROMPT_TEMPLATE, STREAM_PROMPT_TEMPLATE

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
            result = await generate_sql(QU, model, tokenizer, config)
        except Exception as e:
            # 1) generate_sql() 자체가 도중에 예외를 던지는 경우
            print("[ERROR] generate_sql() 도중 예외 발생:", e)
            # 멈추지 않고, 에러 형식으로 데이터를 만들어 반환
            docs = "테이블 조회 시도 중 예외가 발생했습니다. " \
                "해당 SQL을 실행할 수 없어서 테이블 데이터를 가져오지 못했습니다."
            docs_list = []
            return docs, docs_list

        # 2) 함수가 정상 실행됐지만 결과가 None인 경우(= SQL 쿼리 결과가 없거나 오류)
        if result is None:
            print("[WARNING] generate_sql()에서 None을 반환했습니다. " 
                "SQL 수행 결과가 없거나 에러가 발생한 것일 수 있습니다.")
            docs = "테이블 조회 결과가 비어 있습니다. " \
                "조회할 데이터가 없거나 SQL 오류가 발생했습니다."
            docs_list = []
            return docs, docs_list

        # 정상적인 경우(튜플 언패킹)
        final_sql_query, title, explain, table_json, chart_json = result

        # docs : LLM 입력용 (string)
        PROMPT = (
            f"다음은 SQL 추출에 사용된 쿼리문: {final_sql_query}\n\n"
            f"추가 설명: {explain}\n\n"
            f"실제 SQL 추출된 데이터: {str(table_json)}\n\n"
        )

        ### Not Used anymore! ###
        # docs_list : 사용자에게 보여줄 정보(List)
        docs_list = [
            {"title": title, "data": table_json},
            {"title": "시각화 차트", "data": chart_json},
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

### RAG와 다르게 SQL내에서도 vLLM 모델을 사용해야 하므로 따로 정의 ###
@time_tracker
async def execute_sql(QU, KE, TA, TI, **kwargs):
    from SQL_NS import get_metadata, run_sql_unno

    print("[SANGJE]: execute_sql : 진입")
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    embed_model = kwargs.get("embed_model")
    embed_tokenizer = kwargs.get("embed_tokenizer")
    data = kwargs.get("data")
    config = kwargs.get("config")

    metadata_location, metadata_unno = get_metadata(config)
    print(f"✅ Metadata loaded:{metadata_location[:100]}")
    PROMPT =\
f'''
<bos>
<system>
"YourRole": "질문으로 부터 조건을 추출하는 역할",
"YourJob": "아래 요구 사항에 맞추어 'unno', 'class', 'pol_port', 'pod_port' 정보를 추출하여, 예시처럼 답변을 구성해야 합니다.",
"Requirements": [
    unno: UNNO Number는 4개의 숫자로 이루어진 위험물 번호 코드야. 
    class : UN Class는 2.1, 6.0,,, 의 숫자로 이루어진 코드야.
    pol_port, pod_port: 항구 코드는 5개의 알파벳 또는 나라의 경우 2개의 알파벳과 %로 이루어져 있어. 다음은 항구 코드에 대한 메타데이터야 {metadata_location}. 여기에서 매칭되는 코드만을 사용해야 해. 항구는 항구코드, 나라는 2개의 나라코드와 %를 사용해.
    unknown : 질문에서 찾을 수 없는 정보는 NULL을 출력해줘.
]

"Examples": [
    "질문": "UN 번호 1689 화물의 부산에서 미즈시마로의 선적 가능 여부를 확인해 주세요.",
    "답변": "<unno/>1689<unno>\\n<class/>NULL<class>\\n<pol_port/>KRPUS<pol_port>\\n<pod_port/>JPMIZ<pod_port>"

    "질문": "UN 클래스 2.1 화물의 한국에서 일본으로의 선적 가능 여부를 확인해 주세요.",
    "답변": "<unno/>NULL<unno>\\n<class/>2.1<class>\\n<pol_port/>KR%<pol_port>\\n<pod_port/>JP%<pod_port>"
]
- 최종 출력은 반드시 다음 4가지 항목을 포함해야 합니다:
    <unno/>...<unno>
    <class/>...<class>
    <pol_port/>...<pol_port>
    <pod_port/>...<pod_port>
</system>

<user>
질문: "{QU}"
</user>

<assistant>
답변:
</assistant>
'''
    try:
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
            top_k=config.model.top_k,
            top_p=config.model.top_p,
            repetition_penalty=config.model.repetition_penalty,
        )
        accepted_request_id = str(uuid.uuid4())
        outputs_result = await collect_vllm_text(PROMPT, model, sampling_params, accepted_request_id)
        print(f"✅ SQL Model Outputs:{outputs_result}")

        # Regular expression to extract content between <query/> and <query>
        unno_pattern = r'<unno.*?>(.*?)<unno.*?>'
        class_pattern = r'<class.*?>(.*?)<class.*?>'
        pol_port_pattern = r'<pol_port.*?>(.*?)<pol_port.*?>'
        pod_port_pattern = r'<pod_port.*?>(.*?)<pod_port.*?>'

        UN_number = re.search(unno_pattern, outputs_result, re.DOTALL).group(1)
        UN_class = re.search(class_pattern, outputs_result, re.DOTALL).group(1)
        POL = re.search(pol_port_pattern, outputs_result, re.DOTALL).group(1)
        POD = re.search(pod_port_pattern, outputs_result, re.DOTALL).group(1)

        print(f"✅ UN_number:{UN_number}, UN_class:{UN_class}, POL:{POL}, POD:{POD}")
        final_sql_query, result = run_sql_unno(UN_class, UN_number, POL, POD)
        
        ### Temporary ###
        title, explain, table_json, chart_json = (None,) * 4   
        
        result = final_sql_query, title, explain, result, chart_json

    except Exception as e:
        # 1) generate_sql() 자체가 도중에 예외를 던지는 경우
        print("[ERROR] generate_sql() 도중 예외 발생:", e)
        # 멈추지 않고, 에러 형식으로 데이터를 만들어 반환
        docs = "테이블 조회 시도 중 예외가 발생했습니다. " \
            "해당 SQL을 실행할 수 없어서 테이블 데이터를 가져오지 못했습니다."
        docs_list = []
        return docs, docs_list

    # 2) 함수가 정상 실행됐지만 결과가 None인 경우(= SQL 쿼리 결과가 없거나 오류)
    if result is None:
        print("[WARNING] generate_sql()에서 None을 반환했습니다. " 
            "SQL 수행 결과가 없거나 에러가 발생한 것일 수 있습니다.")
        docs = "테이블 조회 결과가 비어 있습니다. " \
            "조회할 데이터가 없거나 SQL 오류가 발생했습니다."
        docs_list = []
        return docs, docs_list

    # 정상적인 경우(튜플 언패킹)
    final_sql_query, title, explain, table_json, chart_json = result

    # docs : LLM 입력용 (string)
    PROMPT = (
        f"다음은 SQL 추출에 사용된 쿼리문: {final_sql_query}\n\n"
        f"추가 설명: {explain}\n\n"
        f"실제 SQL 추출된 데이터: {str(table_json)}\n\n"
    )

    ### Not Used anymore! ###
    # docs_list : 사용자에게 보여줄 정보(List)
    # docs_list = [
    #     {"title": title, "data": table_json},
    #     {"title": "시각화 차트", "data": chart_json},
    # ]
    docs_list = None
    print("[SOOWAN]: execute_rag : 테이블 부분 정상 처리 완료")
    return PROMPT, docs_list

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
        print("##### query_sort is starting, attempt:", attempt+1, "#####")
        
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
        
        # [DEBUG-CHANGE]: Check each match before calling group(1)
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


# @time_tracker
# def sort_by_time(time_bound, data):
#     date_format = "%Y-%m-%d"
#     target_date_start = datetime.strptime(time_bound.split(":")[0], date_format)
#     target_date_end = datetime.strptime(time_bound.split(":")[1], date_format)

#     matching_indices = [
#         i
#         for i, date in enumerate(data["times"])
#         if (not isinstance(date, str)) and (target_date_start < date < target_date_end)
#     ]

#     (
#         data["file_names"],
#         data["titles"],
#         data["times"],
#         data["vectors"],
#         data["texts"],
#         data["texts_short"],
#         data["texts_vis"],
#     ) = (
#         [lst[i] for i in matching_indices]
#         for lst in (
#             data["file_names"],
#             data["titles"],
#             data["times"],
#             data["vectors"],
#             data["texts"],
#             data["texts_short"],
#             data["texts_vis"],
#         )
#     )
#     return data


# @time_tracker
# def retrieve(query, data, N, embed_model, embed_tokenizer):
#     print("[SOOWAN] retrieve : 진입")
#     print("[SOOWAN] retrieve : 진입 정보 :", query)
    
#     sim_score = cal_sim_score(query, data["vectors"], embed_model, embed_tokenizer)
#     print("[SOOWAN] retrieve : sim_score :", sim_score)
    
#     try:
#         bm25_score = cal_bm25_score(query, data["texts_short"], embed_tokenizer)
#     except Exception as e:
#         print("[SOOWAN] retrieve : BM25 score exception, using zeros", e)
#         bm25_score = np.zeros(len(data["texts_short"]))
#     print("[SOOWAN] retrieve : bm25_score")
    
#     scaled_sim_score = min_max_scaling(sim_score)
#     scaled_bm25_score = min_max_scaling(bm25_score)
#     score = scaled_sim_score * 0.4 + scaled_bm25_score * 0.6
#     top_k = score[:, 0, 0].argsort()[-N:][::-1]
#     documents = ""
#     documents_list = []
#     for i, index in enumerate(top_k):
#         documents += f"{i+1}번째 검색자료 (출처:{data['file_names'][index]}) :\n{data['texts_short'][index]}\n"
#         documents_list.append({
#             "file_name": data["file_names"][index],
#             "title": data["titles"][index],
#             "contents": data["texts_vis"][index],
#         })
#         print("\n" + beep)
#     print("-------------자료 검색 성공--------------")
#     return documents, documents_list

@time_tracker
def sort_by_time(time_bound, data):
    """
    원본 데이터는 유지하고 필터링된 복사본을 반환하는 함수
    """
    # 원본 문서 수 로깅
    original_count = len(data["times"])
    print(f"[시간 필터 전] 문서 수: {original_count}")
    
    # "all" 시간 범위 특별 처리
    if time_bound == "all" or time_bound == "1900-01-01:2099-01-01":
        print(f"[시간 필터] 전체 기간 사용 - 모든 문서 포함")
        return data  # 원본 데이터 그대로 반환
    
    # 시간 범위 파싱
    date_format = "%Y-%m-%d"
    target_date_start = datetime.strptime(time_bound.split(":")[0], date_format)
    target_date_end = datetime.strptime(time_bound.split(":")[1], date_format)
    
    # 시간 범위에 맞는 문서 인덱스 찾기
    matching_indices = [
        i
        for i, date in enumerate(data["times"])
        if (not isinstance(date, str)) and (target_date_start < date < target_date_end)
    ]
    
    filtered_count = len(matching_indices)
    print(f"[시간 필터 후] 문서 수: {filtered_count}, 기간: {time_bound}")
    
    # 너무 적은 문서가 남은 경우 경고 로그
    if filtered_count < 50 and filtered_count < original_count * 0.1:
        print(f"[경고] 시간 필터로 인해 문서가 크게 줄었습니다: {original_count} → {filtered_count}")
    
    # 필터링된 데이터를 새로운 딕셔너리에 복사
    filtered_data = {}
    filtered_data["file_names"] = [data["file_names"][i] for i in matching_indices]
    filtered_data["titles"] = [data["titles"][i] for i in matching_indices]
    filtered_data["times"] = [data["times"][i] for i in matching_indices]
    filtered_data["chunk_ids"] = [data["chunk_ids"][i] for i in matching_indices]  # 추가된 부분
    
    # 벡터 타입에 따른 다른 처리
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
        
        # Combined score (0.4 semantic + 0.6 lexical)
        score = scaled_sim_score * 0.4 + scaled_bm25_score * 0.6
        score_values = score[:, 0, 0]
        top_k = score[:, 0, 0].argsort()[-N:][::-1]
        
        # Log top results for debugging
        logging.info(f"Top {N} document indices: {top_k}")
        logging.info(f"Top {N} document scores: {[score[:, 0, 0][i] for i in top_k]}")
        logging.info(f"Top document titles: {[data['titles'][i] for i in top_k]}")
        documents = ""
        documents_list = []
        for i, index in enumerate(top_k):
            score_str = f"{score_values[index]:.4f}"
            documents += f"{i+1}번째 검색자료 (출처:{data['file_names'][index]}) :\n{data['texts_short'][index]}, , Score: {score_str}\n"
            documents_list.append({
                "file_name": data["file_names"][index],
                "title": data["titles"][index],
                "contents": data["texts_vis"][index],
                "chunk_id": data["chunk_ids"][index],
            })
        print("-------------자료 검색 성공--------------")
        print("-------", documents_list, "-------")
        print("---------------------------------------")
        return documents, documents_list
        
        # Continue with document assembly...
    except Exception as e:
        logging.error(f"Retrieval error: {str(e)}", exc_info=True)
        return "", []

@time_tracker
def expand_time_range_if_needed(time_bound, data, min_docs=50):
    """
    시간 필터링 결과가 너무 적은 경우 자동으로 시간 범위를 확장하는 함수
    """
    # "all" 시간 범위는 그대로 사용
    if time_bound == "all" or time_bound == "1900-01-01:2099-01-01":
        print(f"[시간 범위] 전체 기간 사용")
        return data
    
    # 원래 시간 범위로 먼저 시도
    filtered_data = sort_by_time(time_bound, data)
    filtered_count = len(filtered_data.get("times", []))
    
    # 필터링된 문서 수가 충분하면 바로 반환
    if filtered_count >= min_docs:
        print(f"[시간 범위] 원래 범위로 충분한 문서 확보: {filtered_count}개")
        return filtered_data
    
    # 시간 범위 확장 시도
    print(f"[시간 범위 확장] 원래 범위는 {filtered_count}개 문서만 제공 (최소 필요: {min_docs}개)")
    
    # 원래 날짜 파싱
    date_format = "%Y-%m-%d"
    try:
        start_date = datetime.strptime(time_bound.split(":")[0], date_format)
        end_date = datetime.strptime(time_bound.split(":")[1], date_format)
    except Exception as e:
        print(f"[시간 범위 오류] 날짜 형식 오류: {time_bound}, 오류: {e}")
        return data  # 오류 시 원본 데이터 반환
    
    # 점진적으로 더 넓은 범위 시도
    expansions = [
        (3, "3개월"),
        (6, "6개월"),
        (12, "1년"),
        (24, "2년"),
        (60, "5년")
    ]
    
    for months, label in expansions:
        # 양방향으로 균등하게 확장
        new_start = start_date - timedelta(days=30*months//2)
        new_end = end_date + timedelta(days=30*months//2)
        
        new_range = f"{new_start.strftime(date_format)}:{new_end.strftime(date_format)}"
        print(f"[시간 범위 확장] {label} 확장 시도: {new_range}")
        
        expanded_data = sort_by_time(new_range, data)
        expanded_count = len(expanded_data.get("times", []))
        
        if expanded_count >= min_docs:
            print(f"[시간 범위 확장] {label} 확장으로 {expanded_count}개 문서 확보")
            return expanded_data
    
    # 모든 확장이 실패하면 전체 데이터셋 사용
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


# @time_tracker
# def cal_bm25_score(query, indexes, embed_tokenizer):
#     print("[SOOWAN] cal_bm25_score : 진입")
#     try:
#         tokenized_corpus = [
#             embed_tokenizer(
#                 text,
#                 return_token_type_ids=False,
#                 return_attention_mask=False,
#                 return_offsets_mapping=False,
#             )
#             for text in indexes
#         ]
#         tokenized_corpus = [
#             embed_tokenizer.convert_ids_to_tokens(corpus["input_ids"])
#             for corpus in tokenized_corpus
#         ]
#         print(f"[SOOWAN] cal_bm25_score : Tokenized corpus (first 2 items): {tokenized_corpus[:2]}")
#     except Exception as e:
#         print(f"[SOOWAN ERROR BM25] Error tokenizing corpus: {str(e)}")
#         return np.zeros(len(indexes))
#     if not tokenized_corpus or all(len(tokens) == 0 for tokens in tokenized_corpus):
#         print("[SOOWAN] cal_bm25_score: Empty tokenized corpus, returning zeros.")
#         return np.zeros(len(indexes))
#     try:
#         bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
#     except Exception as e:
#         print(f"[SOOWAN ERROR BM25] Error initializing BM25: {str(e)}")
#         return np.zeros(len(indexes))
#     try:
#         tokenized_query = embed_tokenizer(query)
#         tokenized_query = embed_tokenizer.convert_ids_to_tokens(tokenized_query["input_ids"])
#         print(f"[SOOWAN] cal_bm25_score : Tokenized query: {tokenized_query}")
#     except Exception as e:
#         print(f"[SOOWAN ERROR BM25] Error tokenizing query: {str(e)}")
#         return np.zeros(len(indexes))
#     try:
#         bm25_score = bm25.get_scores(tokenized_query)
#         print(f"[SOOWAN] cal_bm25_score : BM25 score: {bm25_score}")
#     except Exception as e:
#         print(f"[SOOWAN ERROR BM25] Error computing BM25 scores: {str(e)}")
#         return np.zeros(len(indexes))
#     return np.array(bm25_score)
@time_tracker
def cal_bm25_score(query, indexes, embed_tokenizer):
    logging.info(f"Starting BM25 calculation for query: {query}")
    logging.info(f"Document count: {len(indexes)}")
    
    if not indexes:
        logging.warning("Empty document list provided to BM25")
        return np.zeros(0)
        
    # Process documents individually to isolate failures
    tokenized_corpus = []
    for i, text in enumerate(indexes):
        try:
            tokens = embed_tokenizer(text, return_token_type_ids=False,
                                    return_attention_mask=False,
                                    return_offsets_mapping=False)
            tokens = embed_tokenizer.convert_ids_to_tokens(tokens["input_ids"])
            if len(tokens) == 0:
                logging.warning(f"Document {i} tokenized to empty list")
                tokens = ["<empty>"]  # Placeholder to avoid BM25 errors
            tokenized_corpus.append(tokens)
        except Exception as e:
            logging.error(f"Failed to tokenize document {i}: {str(e)}")
            tokenized_corpus.append(["<error>"])  # Placeholder
    
    try:
        bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
        tokenized_query = embed_tokenizer.convert_ids_to_tokens(
            embed_tokenizer(query)["input_ids"]
        )
        scores = bm25.get_scores(tokenized_query)
        
        # Check for valid scores
        if np.isnan(scores).any() or np.isinf(scores).any():
            logging.warning("BM25 produced NaN/Inf scores - replacing with zeros")
            scores = np.nan_to_num(scores)
            
        logging.info(f"BM25 scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
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
        input_ids = tokenizer(PROMPT, return_tensors="pt", truncation=True, max_length=4024).to("cuda")
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
        generated_tokens = outputs[0].shape[0]
        answer = tokenizer.decode(outputs[0][token_count:], skip_special_tokens=True)
        print(answer)
        print(">>> decode done, returning answer")
    return answer


@time_tracker
async def collect_vllm_text(PROMPT, model, sampling_params, accepted_request_id):
    import asyncio, concurrent.futures
    print("[SOOWAN] collect_vllm_text 진입 PROMPT: ", PROMPT)
    outputs = []
    async for output in model.generate(PROMPT, request_id=accepted_request_id, sampling_params=sampling_params):
        outputs.append(output)
    if not outputs:
        raise RuntimeError("No outputs were generated by the model.")
    final_output = next((o for o in outputs if getattr(o, "finished", False)), outputs[-1])
    answer = "".join([getattr(comp, "text", "") for comp in getattr(final_output, "outputs", [])])
    print("[SOOWAN] 답변 : ", answer)
    return answer


@time_tracker
async def generate_answer_stream(query, docs, model, tokenizer, config):
    prompt = STREAM_PROMPT_TEMPLATE.format(docs=docs, query=query)
    print("최종 LLM 추론용 prompt 생성 : ", prompt)
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
        async for partial_chunk in collect_vllm_text_stream(prompt, model, sampling_params, request_id):
            # print(f"[STREAM] generate_answer_stream yielded: {partial_chunk}")
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
            # print(f"[STREAM] collect_vllm_text_stream yielding: {completion.text}")
            yield completion.text


if __name__ == "__main__":
    import asyncio
    # engine = AsyncLLMEngine.from_engine_args(engine_args, start_engine_loop=False)
    # if not engine.is_running:
    #     engine.start_background_loop()
    
    async def main():
        status = True
        while status:
            query = input("질문 : ")
            QU, TA, TI = await query_sort({"user_input": query, "model": None, "tokenizer": None, "embed_model": None, "embed_tokenizer": None, "data": None, "config": None})
            print("query_sort result done")
            if TA == "yes":
                print("\n" + beep)
                SQL_results = generate_sql(QU)
                answer = await generate(SQL_results, query)
                print(answer)
                print("\n" + beep)
            else:
                file_names, titles, times, vectors, texts, texts_short = sort_by_time(TI, file_names, titles, times, vectors, texts, texts_short)
                print("\n" + beep)
                docs = retrieve(QU, vectors, texts, texts_short, file_names, N)
                print("\n" + beep)
                answer = await generate(docs, query)
                print(answer)
                print("\n" + beep)
    asyncio.run(main())
