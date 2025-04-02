# core/sql_processing.py
"""
SQL 관련 기능을 제공하는 모듈

주요 기능:
1. 쿼리로부터 SQL 생성 및 변환
2. SQL 실행 및 결과 처리
"""

import logging
import re
import uuid
from typing import Dict, List, Tuple, Any, Optional, Union

from utils.tracking import time_tracker
from core.SQL_NS import run_sql_unno, run_sql_bl, get_metadata
from prompt import SQL_EXTRACTION_PROMPT_TEMPLATE

# 로깅 설정
logger = logging.getLogger("RAG.SQLProcessing")

@time_tracker
async def generate_sql(user_query: str, model, tokenizer, config) -> Optional[Tuple]:
    """
    사용자 쿼리를 분석하여 SQL을 생성하고 실행
    
    Args:
        user_query: 사용자 질문
        model: 언어 모델
        tokenizer: 토크나이저
        config: 설정
        
    Returns:
        Optional[Tuple]: SQL 실행 결과 (실패 시 None)
            - final_sql_query: 실행된 SQL 쿼리
            - title: 쿼리 제목
            - explain: 쿼리 설명
            - table_json: 테이블 데이터
            - chart_json: 차트 데이터
            - detailed_result: 상세 결과
    """
    logger.info("SQL 생성 시작: user_query='%s'", user_query)
    
    # 메타데이터 로드
    metadata_location, metadata_unno = get_metadata(config)
    logger.info("메타데이터 로드 완료")
    
    # 프롬프트 구성
    prompt = SQL_EXTRACTION_PROMPT_TEMPLATE.format(
        metadata_location=metadata_location, 
        query=user_query
    )
    
    # 샘플링 파라미터 설정
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
    
    # 변수 초기화
    un_number = un_class = pol = pod = "NULL"
    unno_pattern = r'<unno.*?>(.*?)<unno.*?>'
    class_pattern = r'<class.*?>(.*?)<class.*?>'
    pol_port_pattern = r'<pol_port.*?>(.*?)<pol_port.*?>'
    pod_port_pattern = r'<pod_port.*?>(.*?)<pod_port.*?>'
    
    while attempt < max_attempts:
        # vLLM으로 추출 실행
        accepted_request_id = str(uuid.uuid4())
        from core.generation import collect_vllm_text
        outputs_result = await collect_vllm_text(prompt, model, sampling_params, accepted_request_id)
        
        logger.info(f"SQL 추출 시도 {attempt+1}: outputs_length={len(outputs_result)}")
        
        # 정규표현식으로 태그 추출
        match_unno = re.search(unno_pattern, outputs_result, re.DOTALL)
        un_number = match_unno.group(1).strip() if match_unno else "NULL"
        
        match_class = re.search(class_pattern, outputs_result, re.DOTALL)
        un_class = match_class.group(1).strip() if match_class else "NULL"
        
        match_pol = re.search(pol_port_pattern, outputs_result, re.DOTALL)
        pol = match_pol.group(1).strip() if match_pol else "NULL"
        
        match_pod = re.search(pod_port_pattern, outputs_result, re.DOTALL)
        pod = match_pod.group(1).strip() if match_pod else "NULL"
        
        logger.info(f"추출 결과: UN_number='{un_number}', UN_class='{un_class}', POL='{pol}', POD='{pod}'")
        
        # 충분한 정보가 추출된 경우 종료
        if ((un_number != "NULL" or un_class != "NULL") and pol != "NULL" and pod != "NULL"):
            break
            
        attempt += 1
    
    logger.info(f"최종 추출 값: UN_number='{un_number}', UN_class='{un_class}', POL='{pol}', POD='{pod}'")
    
    # SQL 실행
    try:
        # DG 가능 여부 확인
        final_sql_query, result = run_sql_unno(un_class, un_number, pol, pod)
        
        # 상세 B/L SQL
        detailed_sql_query, detailed_result = run_sql_bl(un_class, un_number, pol, pod)
        
        # 임시 값 설정 (실제 구현에서는 적절한 값으로 대체)
        title = f"위험물 {un_class} {un_number}의 {pol}에서 {pod}로의 선적 가능 여부"
        explain = f"위험물 클래스 {un_class}, UN 번호 {un_number}의 {pol}에서 {pod}로의 선적 가능 여부를 조회했습니다."
        
        # 데이터 포맷
        table_json = result
        chart_json = None
        
        logger.info("SQL 실행 완료")
        return final_sql_query, title, explain, table_json, chart_json, detailed_result
        
    except Exception as e:
        logger.error(f"SQL 실행 중 오류: {str(e)}", exc_info=True)
        return None