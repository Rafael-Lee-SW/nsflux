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

# 내부 모듈 임포트
from core.retrieval import retrieve, expand_time_range_if_needed
from core.generation import generate, collect_vllm_text, collect_vllm_text_stream
from core.sql_processing import generate_sql

# SQL 관련 함수 임포트
from core.SQL_NS import run_sql_unno, run_sql_bl, get_metadata

# 프롬프트 템플릿 불러오기
from prompt import (
    QUERY_SORT_PROMPT, 
    GENERATE_PROMPT_TEMPLATE, 
    STREAM_PROMPT_TEMPLATE
)

# 유틸리티 임포트
from utils.tracking import time_tracker

# vLLM 관련 임포트
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.model_executor.models.interfaces import SupportsMultiModal

# Tracking
from utils.debug_tracking import get_performance_monitor

# 로깅 설정
logger = logging.getLogger("RAG")

# 글로벌 구분자
SECTION_SEPARATOR = "-" * 100

@time_tracker
async def execute_rag(
    query: str, 
    keywords: str, 
    needs_table: str, 
    time_range: str, 
    **kwargs
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
    embed_model = kwargs.get("embed_model")
    embed_tokenizer = kwargs.get("embed_tokenizer")
    data = kwargs.get("data")
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
            final_sql_query, title, explain, table_json, chart_json, detailed_result = result
            
            # LLM 입력용 프롬프트 구성
            prompt = (
                f"실제 사용된 SQL문: {final_sql_query}\n\n"
                f"추가 설명: {explain}\n\n"
                f"실제 SQL 추출된 데이터: {str(table_json)}\n\n"
                f"실제 선적된 B/L 데이터: {str(detailed_result)}\n\n"
            )
            
            # 결과 메타데이터 구성
            docs_list = [
                {"title": title, "data": table_json},
                {"title": "DG B/L 상세 정보", "data": detailed_result},
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
        
        # 적응형 시간 필터링 적용
        filtered_data = expand_time_range_if_needed(time_range, data, min_docs=50)
        
        # 문서 검색 실행
        logger.info("검색에 사용되는 문서 수: %d", len(filtered_data.get("vectors", [])))
        docs, docs_list = retrieve(
            keywords, 
            filtered_data, 
            config.N, 
            embed_model, 
            embed_tokenizer
        )
        
        return docs, docs_list

@time_tracker
async def generate_answer(
    query: str, 
    docs: str, 
    **kwargs
) -> str:
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
        
        # 프롬프트 구성
        prompt = QUERY_SORT_PROMPT.format(user_query=query)
        logger.info("query_sort 시작 (시도 %d)", attempt + 1)
        
        # LLM에서 응답 생성
        if config.use_vllm:
            sampling_params = SamplingParams(
                max_tokens=config.model.max_new_tokens,
                temperature=config.model.temperature,
                top_k=config.model.top_k,
                top_p=config.model.top_p,
                repetition_penalty=config.model.repetition_penalty,
            )
            request_id = str(uuid.uuid4())
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
            answer = tokenizer.decode(outputs[0][token_count:], skip_special_tokens=True)
        
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
                
            logger.info("질문 구체화 결과: 질문='%s', 키워드='%s', 테이블='%s', 시간='%s'", qu, ke, ta, ti)
            return qu, ke, ta, ti
        else:
            logger.error("필요한 태그가 누락됨. 재시도 %d", attempt + 1)
            attempt += 1
    
    # 최대 시도 횟수를 초과한 경우
    raise ValueError("LLM이 올바른 태그 형식의 답변을 생성하지 못했습니다.")

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
async def generate_answer_stream(
    query: str, 
    docs: str, 
    model, 
    tokenizer, 
    config, 
    http_query: Dict
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
    # 프롬프트 구성
    prompt = STREAM_PROMPT_TEMPLATE.format(docs=docs, query=query)
    logger.info("스트리밍 답변 생성 시작: prompt_length=%d", len(prompt))
    
    # 이미지 처리 관련 파라미터
    image_data = http_query.get("image_data")
    pil_image = None
    
    # 요청 ID 추출 (성능 모니터링 용)
    request_id = http_query.get("page_id") or http_query.get("qry_id")
    if not request_id:
        request_id = str(uuid.uuid4())
    
    # 이미지 데이터가 있는 경우 처리
    if image_data:
        try:
            # 이미지 처리 로직은 image_processing 모듈로 이동
            from core.image_processing import prepare_image
            pil_image = await prepare_image(image_data)
            logger.info("이미지 로드 성공: %s", str(pil_image.size if pil_image else None))
        except Exception as e:
            logger.error("이미지 로드 실패: %s", str(e))
    
    # 스트리밍 방식으로 답변 생성
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
            generate_request = await prepare_multimodal_request(prompt, pil_image, config.model_id, tokenizer)
        else:
            generate_request = prompt
        
        # 성능 모니터링 위한 초기화
        from utils.debug_tracking import get_performance_monitor
        perf_monitor = get_performance_monitor()
        if request_id:
            perf_monitor.update_request(
                request_id, 0, 
                checkpoint="start_vllm_generation",
                current_output=""
            )
        
        # 스트리밍 생성
        async for partial_chunk in collect_vllm_text_stream(generate_request, model, sampling_params, request_id):
            # 성능 모니터링 업데이트는 collect_vllm_text_stream 내부에서 수행
            yield partial_chunk
    else:
        # HuggingFace 모델 사용 (비 vLLM)
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

# 이 RAG.py 파일은 이제 다른 모듈에서 필요한 함수들만 노출하며,
# 세부 구현은 각각의 특화된 모듈로 이동되었습니다.