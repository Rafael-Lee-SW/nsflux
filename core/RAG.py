# core/RAG.py
"""
RAG(Retrieval-Augmented Generation) 시스템의 메인 모듈

이 모듈은 다음 핵심 기능들을 제공합니다:
1. 질문 분류 및 구체화 (query_sort, specific_question)
2. RAG 실행 및 조정 (execute_rag)
3. 응답 생성 (generate_answer, generate_answer_stream)

기타 세부 기능들은 하위 모듈로 분리되었습니다:
- retrieval: 문서 검색 관련 기능
- generation: 텍스트 생성 관련 기능
- image_processing: 이미지 처리 관련 기능
- query_processing: 쿼리 처리 관련 기능
- sql_processing: SQL 쿼리 관련 기능
"""

import asyncio
import logging
import re
import uuid
from typing import Dict, List, Tuple, Any, Optional, Union, Generator
import httpx
from config import config
import textwrap
# 내부 모듈 임포트
from core.retrieval import retrieve, expand_time_range_if_needed # Ax100 API로 대체됨
from core.generation import generate, collect_vllm_text, collect_vllm_text_stream
from core.sql_processing import generate_sql

# SQL 관련 함수 임포트
from core.SQL_NS import run_sql_unno, run_sql_bl, get_metadata

# 프롬프트 템플릿 불러오기
from prompt import (
    QUERY_SORT_PROMPT,
    GENERATE_PROMPT_TEMPLATE,
    STREAM_PROMPT_TEMPLATE,
    IMAGE_DESCRIPTION_PROMPT,
    NON_RAG_PROMPT_TEMPLATE,
    IMAGE_PROMPT_TEMPLATE,
    TABLE_PROMPT_TEMPLATE,
)

# 유틸리티 임포트
from utils.tracking import time_tracker

# vLLM 관련 임포트
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.model_executor.models.interfaces import SupportsMultiModal

# 로깅 설정
logger = logging.getLogger("RAG")

# 글로벌 구분자
SECTION_SEPARATOR = "-" * 100

# RAG 서비스 API 호출 설정
RAG_API_TIMEOUT = 300.0  # 타임아웃 시간 (초)
# RAG_API_MAX_RETRIES = 3  # 최대 재시도 횟수
# RAG_API_RETRY_DELAY = 1.0  # 재시도 간 대기 시간 (초)


@time_tracker
async def execute_rag(
    query: str, keywords: str, needs_table: str, time_range: str, **kwargs
) -> Tuple[str, List[Dict]]:
    """
    RAG(Retrieval-Augmented Generation) 실행의 메인 진입점

    Args:
        query: 구체화된 사용자 질문
        keywords: 검색에 사용할 키워드
        needs_table: 테이블 데이터 필요 여부 ("yes" 또는 "no")
        time_range: 검색 시간 범위 ("all" 또는 "시작일:종료일" 형식)
        **kwargs: 추가 파라미터 (model, tokenizer, embed_model, embed_tokenizer, data, config)

    Returns:
        Tuple[str, List[Dict]]: (검색된 문서 내용, 문서 메타데이터 리스트)
    """
    logger.info("execute_rag 진입: query='%s', needs_table=%s", query, needs_table)

    # 필수 파라미터 추출
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    config = kwargs.get("config")

    # 테이블 필요 여부 확인
    if needs_table == "yes":
        logger.info("테이블 데이터 필요: SQL 생성 시작")
        try:
            # SQL 생성 및 실행
            result = await generate_sql(query, model, tokenizer, config)

            # SQL 실행 결과가 없는 경우
            if result is None:
                logger.warning("SQL 실행 결과 없음")
                docs = "테이블 조회 결과가 비어 있습니다. 조회할 데이터가 없거나 SQL 오류가 발생했습니다."
                docs_list = []
                return docs, docs_list

            # SQL 실행 결과 처리
            final_sql_query, title, explain, table_json, chart_json, detailed_result = (
                result
            )

            # LLM 입력용 프롬프트 구성
            prompt = (
                # f"실제 사용된 SQL문: {final_sql_query}\n\n"
                f"추가 설명: {explain}\n\n"
                f"실제 SQL 추출된 데이터: {str(table_json)}\n\n"
                f"실제 선적된 B/L 데이터: {str(detailed_result)}\n\n"
            )

            # 결과 메타데이터 구성 - RAG 서비스 API 형식에 맞게 수정
            docs_list = [
                {
                    "file_name": "SQL_Result",
                    "title": title,
                    "data": table_json,                 # ← 바로 data 로!
                    "chunk_id": 0,
                },
                {
                    "file_name": "B/L_Detail",
                    "title": "DG B/L 상세 정보",
                    "data": detailed_result,            # ← 바로 data 로!
                    "chunk_id": 1,
                },
            ]

            logger.info("테이블 데이터 처리 완료")
            return prompt, docs_list

        except Exception as e:
            logger.error("SQL 처리 중 오류 발생: %s", str(e), exc_info=True)
            docs = f"테이블 조회 시도 중 예외가 발생했습니다. 해당 SQL을 실행할 수 없어서 테이블 데이터를 가져오지 못했습니다. 오류: {str(e)}"
            docs_list = []
            return docs, docs_list
    else:
        logger.info("표준 검색 실행: 키워드='%s', 시간 범위='%s'", keywords, time_range)

        try:
            async with httpx.AsyncClient(timeout=RAG_API_TIMEOUT) as client:
                # Dual Query Retrieval 전략 구현
                # 1. 첫 번째 쿼리: keywords로 검색 (query_sort가 생성한 구조화된 키워드)
                logger.info("RAG 서비스 API 1차 호출 (keywords)")
                response1 = await client.post(
                    f"{config.rag_service_url}/api/retrieve",
                    json={
                        "query": keywords,
                        "top_n": config.N,
                        "time_bound": time_range,
                        "min_docs": 50,
                    },
                )
                
                if response1.status_code != 200:
                    logger.error(
                        "RAG 서비스 API 1차 호출 오류: %s - %s",
                        response1.status_code,
                        response1.text,
                    )
                    return "검색 서비스에 연결할 수 없습니다.", []
                
                result1 = response1.json()
                docs_list1 = result1.get("documents_list", [])
                
                # 2. 두 번째 쿼리: 원본 query로 검색 (사용자의 원래 질문)
                logger.info("RAG 서비스 API 2차 호출 (원본 query)")
                response2 = await client.post(
                    f"{config.rag_service_url}/api/retrieve",
                    json={
                        "query": query,
                        "top_n": config.N,
                        "time_bound": time_range,
                        "min_docs": 50,
                    },
                )
                
                if response2.status_code != 200:
                    logger.error(
                        "RAG 서비스 API 2차 호출 오류: %s - %s",
                        response2.status_code,
                        response2.text,
                    )
                    # 첫 번째 결과라도 사용
                    docs = result1.get("documents", "")
                    logger.info("1차 호출 결과만 사용: %d개 문서", len(docs_list1))
                    return docs, docs_list1
                
                result2 = response2.json()
                docs_list2 = result2.get("documents_list", [])
                
                # 3. 결과 통합 (chunk_id 기준으로 중복 제거)
                all_docs_list = docs_list1 + docs_list2
                unique_docs_list = []
                seen_chunk_ids = set()
                
                for doc in all_docs_list:
                    chunk_id = doc.get("chunk_id")
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        unique_docs_list.append(doc)
                
                # 두 쿼리 결과의 통계 로깅
                logger.info("Dual Query 결과: keywords=%d개, query=%d개, 통합=%d개 문서", 
                           len(docs_list1), len(docs_list2), len(unique_docs_list))
                
                # keyword 결과와 query 결과의 chunk_id 비교
                keyword_chunk_ids = {doc.get("chunk_id") for doc in docs_list1}
                query_chunk_ids = {doc.get("chunk_id") for doc in docs_list2}
                common_chunks = keyword_chunk_ids.intersection(query_chunk_ids)
                keyword_only = keyword_chunk_ids - query_chunk_ids
                query_only = query_chunk_ids - keyword_chunk_ids
                
                logger.info("공통 chunk_id: %s", common_chunks)
                logger.info("keywords만의 chunk_id: %s", keyword_only)
                logger.info("query만의 chunk_id: %s", query_only)
                
                # Here is the place of new function that delete the not related docs things from docs list
                
                # 통합된 문서 목록을 사용하여 docs 문자열 생성
                docs_list = unique_docs_list
                
                # docs_filter 함수로 필터링
                docs_list = await docs_filter(query, docs_list, model, tokenizer, config)
                logger.info("docs_filter 이후: %d개 문서로 축소", len(docs_list))
                
                # docs 문자열 재조합 (LLM 입력용)
                docs = "\n\n".join(
                    f"[{doc.get('title', '제목없음')}] {_get_doc_text(doc)}"
                    for doc in docs_list
                )
                
                return docs, docs_list

        except httpx.ReadTimeout as e:
            logger.error("RAG 서비스 API 타임아웃: %s", e)
            return "검색 서비스에 연결할 수 없습니다. (타임아웃)", []
        except Exception as e:
            logger.error("RAG 서비스 API 호출 중 오류: %s", e, exc_info=True)
            return f"검색 서비스에 연결할 수 없습니다. 오류: {str(e)}", []

@time_tracker
async def generate_answer(query: str, docs: str, **kwargs) -> str:
    """
    검색된 문서를 기반으로 최종 답변 생성

    Args:
        query: 사용자 질문
        docs: 검색된 문서 내용
        **kwargs: 추가 파라미터 (model, tokenizer, config)

    Returns:
        str: 생성된 답변
    """
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    config = kwargs.get("config")

    answer = await generate(docs, query, model, tokenizer, config)
    return answer


@time_tracker
async def query_sort(params: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    사용자 질문을 분석하여 구체화하고 RAG 파라미터를 추출

    Args:
        params: 파라미터 딕셔너리 (user_input, model, tokenizer, embed_model, embed_tokenizer, data, config)

    Returns:
        Tuple[str, str, str, str]: (구체화된 질문, 키워드, 테이블 필요 여부, 시간 범위)
    """
    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        # 필요 파라미터 추출
        query = params["user_input"]
        model = params["model"]
        tokenizer = params["tokenizer"]
        config = params["config"]
        request_id = params["request_id"]
        
        # 프롬프트 구성
        prompt = QUERY_SORT_PROMPT.format(user_query=query)
        logger.info("query_sort 시작 (시도 %d)", attempt + 1)

        try:
            # LLM에서 응답 생성
            if config.use_vllm:
                sampling_params = SamplingParams(
                    max_tokens=config.model.max_new_tokens,
                    temperature=config.model.temperature,
                    top_k=config.model.top_k,
                    top_p=config.model.top_p,
                    repetition_penalty=config.model.repetition_penalty,
                )
                answer = await collect_vllm_text(prompt, model, sampling_params, request_id)
            else:
                input_ids = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=4024
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
                answer = tokenizer.decode(
                    outputs[0][token_count:], skip_special_tokens=True
                )

            logger.debug("Generated answer: %s", answer)

            # 응답에서 태그로 감싸진 정보 추출
            query_pattern = r"<query.*?>(.*?)<query.*?>"
            keyword_pattern = r"<keyword.*?>(.*?)<keyword.*?>"
            table_pattern = r"<table.*?>(.*?)<table.*?>"
            time_pattern = r"<time.*?>(.*?)<time.*?>"

            m_query = re.search(query_pattern, answer, re.DOTALL)
            m_keyword = re.search(keyword_pattern, answer, re.DOTALL)
            m_table = re.search(table_pattern, answer, re.DOTALL)
            m_time = re.search(time_pattern, answer, re.DOTALL)

            # 모든 필수 태그가 존재하는 경우
            if m_query and m_keyword and m_table and m_time:
                qu = m_query.group(1).strip()
                ke = m_keyword.group(1).strip()
                ta = m_table.group(1).strip()
                ti = m_time.group(1).strip()

                # 'all' 시간 범위 처리
                if ti == "all":
                    ti = "1900-01-01:2099-01-01"

                logger.info(
                    "질문 구체화 결과: 질문='%s', 키워드='%s', 테이블='%s', 시간='%s'",
                    qu,
                    ke,
                    ta,
                    ti,
                )
                return qu, ke, ta, ti
            else:
                logger.error("필요한 태그가 누락됨. 재시도 %d", attempt + 1)
                attempt += 1

        except Exception as e:
            logger.error("질문 구체화 중 오류 발생: %s", str(e), exc_info=True)
            attempt += 1

    # 최대 시도 횟수를 초과한 경우 기본값 반환
    default_qu = params["user_input"]
    default_ke = params["user_input"]
    default_ta = "no"
    default_ti = "1901-01-01:2099-12-01"
    logger.warning(
        "최대 시도 횟수를 초과하여 기본값 반환: 질문='%s', 키워드='%s', 테이블='%s', 시간='%s'",
        default_qu,
        default_ke,
        default_ta,
        default_ti,
    )
    return default_qu, default_ke, default_ta, default_ti

@time_tracker
async def specific_question(params: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    대화 이력을 고려한 구체적인 질문 생성
    (query_sort와 유사하지만 대화 이력 처리에 차이가 있을 수 있음)

    Args:
        params: 파라미터 딕셔너리

    Returns:
        Tuple[str, str, str, str]: (구체화된 질문, 키워드, 테이블 필요 여부, 시간 범위)
    """
    # query_sort와 동일한 로직 사용
    return await query_sort(params)

@time_tracker
async def docs_filter(
    query: str,
    docs_list: List[Dict],
    model,
    tokenizer,
    config,
    max_chars_per_doc: int = 100000,
) -> List[Dict]:
    """
    검색된 문서들 중에서 **사용자 질문과 직접적으로 관련** 있는 문서의 chunk_id만 필터링합니다.
    1) 각 문서를 요약해 리스트 생성
    2) Transformer의 apply_chat_template로 대화형 프롬프트 구성
    3) LLM 호출해 관련 chunk_id만 반환하도록 지시
    4) 응답에서 숫자(id) 파싱 → 필터링
    5) 실패 시 원본 리스트 반환
    """
    if not docs_list:
        return docs_list

    # (1) 문서 요약
    formatted_docs = []
    for doc in docs_list:
        raw = ""
        try:
            raw = _get_doc_text(doc)
        except Exception:
            pass
        summary = textwrap.shorten(raw, width=max_chars_per_doc, placeholder="…")
        formatted_docs.append(
            f"{doc['chunk_id']}. 제목: {doc.get('title', '제목없음')}\n   내용: {summary}"
        )

    # (2) chat 형식 프롬프트 생성
    from transformers import AutoTokenizer
    chat_tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        use_fast=True,
        trust_remote_code=True,
    )
    
    # 시스템 프롬프트 정의
    SYSTEM_PROMPT = """
아래 용어 설명과 유의 사항을 참고하여
**사용자 질문에 직접적으로 필요한 문서**의 chunk_id만 고르십시오.

────────────────────────────────────────
# 용어 설명
- Commission (커미션): 에이전트(대리점)에게 지급되는 모든 금전적 항목
  · 예: Booking Commission, Husbanding Fee, Documentation Fee
- SLC (Container Seal Charge, Tariff of Seal charge):
  • 컨테이너 봉인 요금이면서 **Commission 항목** 중 하나임
  • 남성해운과의 맺은 계약을 맺은 대리점의 커미션 및 요율 포함됨
- D/O (Delivery Order): 화물 인도 지시 수수료
- HK agent: Namsung Shipping Hongkong Ltd, 홍콩 대리점과 동등어

# 유의 사항
1. 질문에 특정 대리점이 언급된 경우, **오직 그 대리점과 관련된 문서만** 선택하세요.
   예: "홍콩 대리점" 질문에는 싱가폴, 타이완 등 다른 대리점 문서를 제외해야 합니다.
2. 질문에 언급된 대리점과 일치하지 않는 문서는 어떤 경우에도 선택하지 마세요.
3. *어떤 문서*도 커미션 표 변경·중단 내역이라면 반드시 포함하되, 언급된 대리점과 관련된 것만 선택하세요.
4. 문서가 여러 개 중복되면 하나만 선택하세요.

# 출력 형식
`35,974,2852` 처럼 **숫자와 콤마만** 한 줄로 출력한다.
"""

    system_content = SYSTEM_PROMPT
    
    # 실제 반환용 문서 목록 생성을 위한 프롬프트 생성
    messages = [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": (
                f"[사용자 질문]\n{query}\n\n[문서 목록]\n" + "\n".join(formatted_docs)
            ),
        },
    ]
    prompt = chat_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # (3) LLM 호출
    sampling_params = SamplingParams(
        max_tokens = 64,
        temperature = 0.1,  # 더 결정론적인 결과를 위해 낮춤
        top_k = 5,         # 후보군을 더 제한
        top_p = 0.9,       # 확률 분포 제한
        repetition_penalty = 1.1  # 반복 방지
    )
    try:
        answer = await collect_vllm_text(
            prompt,
            model,
            sampling_params,
            str(uuid.uuid4()),
        )
        # (4) 숫자 추출 및 필터링
        keep_ids = set(map(int, re.findall(r"\d+", answer)))
        filtered = [doc for doc in docs_list if doc["chunk_id"] in keep_ids]
        
        # # 원래 선택된 문서와 필터링된 문서의 ID를 로깅
        # original_ids = [doc["chunk_id"] for doc in docs_list]
        # filtered_ids = [doc["chunk_id"] for doc in filtered]
        # logger.info(f"원래 선택된 문서 ID: {original_ids}")
        # logger.info(f"필터링 후 문서 ID: {filtered_ids}")
        
        # # 왜 선택했는지 이유를 알기 위해 쓰는 프롬프트
        # messages_reason = [
        #     {
        #         "role": "system",
        #         "content": (
        #             "당신은 문서 선택의 전문가입니다. 다음 지시사항을 따라 각 문서가 선택된 이유를 상세히 설명해주세요.\n\n"
        #             "1. 원래 선택된 모든 문서와 필터링된 문서를 비교 분석하세요.\n"
        #             "2. 각 문서(chunk_id)가 선택되거나 제외된 구체적인 이유를 설명하세요.\n"
        #             "3. 문서의 내용과 사용자 질문 간의 연관성을 명확히 설명하세요.\n"
        #             "4. 특히 다음 용어와 관련된 내용이 있다면 반드시 언급하세요:\n"
        #             "   - SLC (Container Seal Charge): 컨테이너 봉인 요금이면서 커미션 항목\n"
        #             "   - D/O (Delivery Order): 화물인도지시서\n"
        #             "   - HK agent: 홍콩 대리점\n\n"
        #             "5. 대리점 관련 질문인 경우, 해당 대리점과의 연관성을 특별히 강조하세요.\n\n"
        #             "출력 형식:\n"
        #             "1. 원래 선택된 문서 분석:\n"
        #             "   문서 ID: [chunk_id]\n"
        #             "   선택 이유: [상세한 설명]\n"
        #             "   ---\n"
        #             "2. 필터링 결과 분석:\n"
        #             "   최종 선택된 문서 ID: [chunk_id]\n"
        #             "   선택 이유: [상세한 설명]\n"
        #             "   제외된 문서 ID: [chunk_id]\n"
        #             "   제외 이유: [상세한 설명]\n"
        #             "   ---\n"
        #         ),
        #     },
        #     {
        #         "role": "user",
        #         "content": (
        #             f"[사용자 질문]\n{query}\n\n"
        #             f"[원래 선택된 문서 목록]\n" + "\n".join(formatted_docs) + "\n\n"
        #             f"[필터링된 문서 목록]\n" + "\n".join([f"{doc['chunk_id']}. {doc.get('title', '제목없음')}" for doc in filtered])
        #         ),
        #     },
        # ]
        
        # prompt_reason = chat_tokenizer.apply_chat_template(
        #     messages_reason,
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )
        
        # sampling_params_reason = SamplingParams(
        #     max_tokens = 4096,
        #     temperature = 1.0,
        #     top_k = 30,
        #     top_p = 1.0
        # )
        
        # try:
        #     answer_reason = await collect_vllm_text(
        #         prompt_reason,
        #         model,
        #         sampling_params_reason,
        #         str(uuid.uuid4()),
        #     )
        #     logger.info(f"문서 선택 이유 분석: {answer_reason}")
        # except Exception as e:
        #     logger.error("문서 선택 이유 분석 중 오류 발생: %s", e, exc_info=True)
        #     return docs_list
        
        return filtered or docs_list
    except Exception as e:
        logger.error("docs_filter failed, returning original list: %s", e, exc_info=True)
        return docs_list

# core/RAG.py ─ import 아래 아무 곳 (기존 _extract_plain_text 제거 후 ↓ 붙여넣기)
def _get_doc_text(doc: Dict[str, Any]) -> str:
    """
    검색 서버가 내려준 단일 문서(dict)에서 사람이 읽을 텍스트를 추출한다.
    ① doc["text_short"]가 있으면 → 그대로 반환
    ② fallback: doc["contents"] 구조를 평탄화
    ③ 그래도 없으면 빈 문자열
    """
    # 1) text_short 최우선
    short = doc.get("text_short")
    if short:
        return str(short).strip()

    # 2) 기존 contents 호환 (dict / list / str 어떤 형태든 OK)
    contents = doc.get("contents")
    if contents is None:
        return ""

    # 2‑1) str
    if isinstance(contents, str):
        return contents.strip()

    # 2‑2) dict
    if isinstance(contents, dict):
        for key in ("data", "text", "page_content", "content", "body"):
            if key in contents and contents[key]:
                return str(contents[key]).strip()
        return " ".join(str(v).strip() for v in contents.values())

    # 2‑3) list (재귀)
    if isinstance(contents, list):
        return " ".join(_get_doc_text({"contents": c}) for c in contents).strip()

    # 2‑4) 기타
    return str(contents).strip()


@time_tracker
async def generate_answer_stream(
    query: str, docs: str, model, tokenizer, config, http_query: Dict
) -> Generator[str, None, None]:
    """
    스트리밍 방식으로 답변 생성

    Args:
        query: 사용자 질문
        docs: 검색된 문서 내용
        model: 언어 모델
        tokenizer: 토크나이저
        config: 설정
        http_query: HTTP 요청 정보

    Yields:
        str: 생성된 부분 텍스트
    """
    logger.info(
        "[GENERATE_ANSWER_STREAM] ---------- 스트리밍 답변 생성 시작 ----------"
    )

    # -------------------------------------------------------------------------
    # 1) 분기 처리를 위한 파라미터 준비
    # -------------------------------------------------------------------------
    # 이미지가 있는지 여부
    image_data = http_query.get("image_data")
    is_image = bool(image_data)  # image_data가 None이 아니면 True

    # RAG 사용 여부
    use_rag = http_query.get("use_rag", True)

    # -------------------------------------------------------------------------
    # 2) 분기 로직에 따른 프롬프트 선택
    # -------------------------------------------------------------------------
    if is_image and not use_rag:
        # 이미지 있음 + RAG 미사용
        prompt = IMAGE_PROMPT_TEMPLATE.format(query=query)
    elif is_image and use_rag:
        # 이미지 있음 + RAG 사용
        prompt = STREAM_PROMPT_TEMPLATE.format(docs=docs, query=query)
    elif (not is_image) and use_rag:
        # 이미지 없음 + RAG 사용
        prompt = STREAM_PROMPT_TEMPLATE.format(docs=docs, query=query)
    else:
        # 이미지 없음 + RAG 미사용
        prompt = NON_RAG_PROMPT_TEMPLATE.format(query=query)

    logger.info(f"[PROMPT SELECTION] is_image={is_image}, use_rag={use_rag}")
    logger.debug(f"[PROMPT CONTENT]\n{prompt}")

    # -------------------------------------------------------------------------
    # 3) request_id 설정 (성능 모니터링 등 용도)
    # -------------------------------------------------------------------------
    request_id = http_query.get("page_id") or http_query.get("qry_id")
    if not request_id:
        request_id = str(uuid.uuid4())

    # -------------------------------------------------------------------------
    # 4) 실제 이미지 처리가 필요한 경우, 이미지 로드 (core.image_processing 사용)
    # -------------------------------------------------------------------------
    pil_image = None
    if image_data:
        try:
            # 이미지 처리 로직은 image_processing 모듈로 이동
            from core.image_processing import prepare_image

            pil_image = await prepare_image(image_data)
            logger.info(
                "이미지 로드 성공: %s", str(pil_image.size if pil_image else None)
            )
        except Exception as e:
            logger.error("이미지 로드 실패: %s", str(e))

    # -------------------------------------------------------------------------
    # 5) 스트리밍 텍스트 생성 (vLLM 또는 HF)
    # -------------------------------------------------------------------------
    if config.use_vllm:
        sampling_params = SamplingParams(
            max_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
            top_k=config.model.top_k,
            top_p=config.model.top_p,
            repetition_penalty=config.model.repetition_penalty,
        )

        # 멀티모달 요청 구성
        if pil_image:
            # 멀티모달 처리를 위한 요청 구성
            from core.image_processing import prepare_multimodal_request

            generate_request = await prepare_multimodal_request(
                prompt, pil_image, config.model_id, tokenizer
            )
        else:
            generate_request = prompt

        # 스트리밍 생성
        async for partial_chunk in collect_vllm_text_stream(
            generate_request, model, sampling_params, request_id
        ):
            # 성능 모니터링 업데이트는 collect_vllm_text_stream 내부에서 수행
            yield partial_chunk
    else:
        # HuggingFace 모델 사용 (Not use vLLM)
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

        import threading

        t = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        t.start()

        for new_token in streamer:
            yield new_token


# 이 RAG.py 파일은 이제 다른 모듈에서 필요한 함수들만 노출하며,
# 세부 구현은 각각의 특화된 모듈로 이동되었습니다.
