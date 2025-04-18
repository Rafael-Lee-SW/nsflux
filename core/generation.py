# core/generation.py
"""
텍스트 생성 관련 기능을 제공하는 모듈

주요 기능:
1. LLM을 통한 텍스트 생성
2. 스트리밍 방식의 텍스트 생성
3. vLLM 관련 유틸리티 기능
"""

import logging
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Generator, AsyncGenerator
import uuid

from utils.tracking import time_tracker
from prompt import (
    GENERATE_PROMPT_TEMPLATE,
    NON_RAG_PROMPT_TEMPLATE,
    IMAGE_PROMPT_TEMPLATE,
    TABLE_PROMPT_TEMPLATE,
)

# vLLM 관련 임포트
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams

# 로깅 설정
logger = logging.getLogger("RAG.Generation")


@time_tracker
async def generate(docs: str, query: str, model, tokenizer, config, http_query=None) -> str:
    """
    검색된 문서와 질문을 바탕으로 응답 생성

    Args:
        docs: 검색된 문서 내용
        query: 사용자 질문
        model: 언어 모델
        tokenizer: 토크나이저
        config: 설정
        http_query: HTTP 요청 정보 (선택적)

    Returns:
        str: 생성된 텍스트
    """
    # 테이블 데이터 사용 여부 확인
    use_table = False
    if http_query and "use_table" in http_query:
        use_table = http_query.get("use_table", False)
    
    # 프롬프트 구성
    if use_table:
        # 테이블 데이터 기반 프롬프트 (SQL 결과 포함)
        prompt = TABLE_PROMPT_TEMPLATE.format(
            table_data=docs, 
            docs="", 
            query=query
        )
        logger.info("SQL 테이블 데이터 기반 프롬프트 사용")
    else:
        # 일반 문서 기반 프롬프트
        prompt = GENERATE_PROMPT_TEMPLATE.format(docs=docs, query=query)
    
    logger.info("텍스트 생성 시작: prompt_length=%d, use_table=%s", len(prompt), use_table)

    # vLLM 모드인 경우
    if config.use_vllm:
        from vllm import SamplingParams

        # 샘플링 파라미터 설정
        sampling_params = SamplingParams(
            max_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
            top_k=config.model.top_k,
            top_p=config.model.top_p,
            repetition_penalty=config.model.repetition_penalty,
        )

        # 요청 ID 생성
        accepted_request_id = str(uuid.uuid4())

        # vLLM으로 텍스트 생성
        answer = await collect_vllm_text(
            prompt, model, sampling_params, accepted_request_id
        )

    # 표준 HuggingFace 모델 사용
    else:
        # 입력 토큰화
        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4024
        ).to("cuda")

        # 토큰 수 계산
        token_count = input_ids["input_ids"].shape[1]

        # 텍스트 생성
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

        # 생성된 텍스트만 추출
        answer = tokenizer.decode(outputs[0][token_count:], skip_special_tokens=True)
        logger.debug("생성된 답변: %s", answer)

    logger.info("텍스트 생성 완료")
    return answer


@time_tracker
async def collect_vllm_text(
    prompt: Any, model: AsyncLLMEngine, sampling_params: SamplingParams, request_id: str
) -> str:
    """
    vLLM 모델에서 텍스트 생성 후 결과 수집

    Args:
        prompt: 프롬프트 (텍스트 또는 멀티모달 요청 객체)
        model: vLLM 엔진
        sampling_params: 샘플링 파라미터
        request_id: 요청 ID

    Returns:
        str: 생성된 텍스트
    """
    logger.info(f"vLLM 텍스트 생성 시작: request_id={request_id}")

    outputs = []
    async for output in model.generate(
        prompt, request_id=request_id, sampling_params=sampling_params
    ):
        outputs.append(output)

    # 출력이 없는 경우 오류 발생
    if not outputs:
        error_msg = "모델이 출력을 생성하지 않음"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 마지막 또는 완료된 출력 선택
    final_output = next(
        (o for o in outputs if getattr(o, "finished", False)), outputs[-1]
    )

    # 텍스트 추출
    answer = "".join(
        [getattr(comp, "text", "") for comp in getattr(final_output, "outputs", [])]
    )

    logger.info(f"vLLM 텍스트 생성 완료: text_length={len(answer)}")
    return answer


@time_tracker
async def collect_vllm_text_stream(
    prompt: Any,
    engine: AsyncLLMEngine,
    sampling_params: SamplingParams,
    request_id: str,
) -> AsyncGenerator[str, None]:
    """
    vLLM 모델에서 스트리밍 방식으로 텍스트 생성 및 반환

    Args:
        prompt: 프롬프트 (텍스트 또는 멀티모달 요청 객체)
        engine: vLLM 엔진
        sampling_params: 샘플링 파라미터
        request_id: 요청 ID

    Yields:
        str: 생성된 부분 텍스트
    """
    logger.info(f"vLLM 스트리밍 텍스트 생성 시작: request_id={request_id}")

    try:
        async for request_output in engine.generate(prompt, request_id=request_id, sampling_params=sampling_params):
            
            if not request_output.outputs:
                continue

            for completion in request_output.outputs:
                yield completion.text
                
    except Exception as e:
        logger.error(f"vLLM 스트리밍 생성 중 오류: {str(e)}")
        raise

    logger.info(f"vLLM 스트리밍 텍스트 생성 완료: request_id={request_id}")
