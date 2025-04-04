# prompt_testing.py
"""
프롬프트 테스트 전용 모듈

이 모듈은 RAG 기능 없이 프롬프트만 테스트할 수 있는 기능을 제공합니다.
- 새 프롬프트를 테스트할 수 있음
- 텍스트 및 이미지 입력 지원
- ChatML 형식의 프롬프트 템플릿 적용
- vLLM을 통한 생성
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, Union
from PIL import Image

# 유틸리티 임포트
from utils.tracking import time_tracker
from core.image_processing import prepare_image

# vLLM 관련 임포트
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine

# 로깅 설정
logger = logging.getLogger("PROMPT_TESTING")

@time_tracker
async def test_prompt_with_image(
    system_prompt: str,
    user_text: str,
    image_data: Optional[Union[str, bytes]],
    model: AsyncLLMEngine,
    tokenizer: Any,
    config: Any,
    request_id: str = None
) -> str:
    """
    이미지와 함께 프롬프트를 테스트합니다.
    
    Args:
        system_prompt: 테스트할 시스템 프롬프트
        user_text: 사용자 입력 텍스트
        image_data: 이미지 데이터 (base64 또는 바이너리)
        model: vLLM 엔진
        tokenizer: 토크나이저
        config: 설정
        request_id: 요청 ID
        
    Returns:
        str: 생성된 전체 텍스트
    """
    if not request_id:
        request_id = str(uuid.uuid4())
    
    logger.info(f"프롬프트 테스트 시작: request_id={request_id}")
    
    # 이미지 처리
    pil_image = None
    if image_data:
        try:
            pil_image = await prepare_image(image_data)
            logger.info(f"이미지 로드 성공: {pil_image.size if pil_image else None}")
        except Exception as e:
            logger.error(f"이미지 로드 실패: {str(e)}")
            return f"이미지 로드 오류: {str(e)}"
    
    try:
        # 멀티모달 메시지 설정
        if pil_image:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                config.model_id,
                use_fast=False
            )
            
            # 시스템 메시지와 사용자 메시지 구성
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": pil_image},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
            
            # 채팅 템플릿 적용
            prompt_string = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # 멀티모달 요청 구성
            generate_request = {
                "prompt": prompt_string,
                "multi_modal_data": {
                    "image": [pil_image]
                }
            }
        else:
            # 이미지 없는 경우, 일반 텍스트 프롬프트 구성
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id,
                use_fast=True,
                trust_remote_code=True
            )
            
            # 시스템 메시지와 사용자 메시지 구성
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
            
            # 채팅 템플릿 적용
            prompt_string = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 일반 텍스트 요청
            generate_request = prompt_string
        
        # 샘플링 파라미터 설정
        sampling_params = SamplingParams(
            max_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
            top_k=config.model.top_k,
            top_p=config.model.top_p,
            repetition_penalty=config.model.repetition_penalty,
        )
        
        # vLLM으로 생성
        logger.info("모델 생성 시작")
        result_chunks = []
        
        async for output in model.generate(
            prompt=generate_request,
            sampling_params=sampling_params,
            request_id=request_id
        ):
            result_chunks.append(output)
        
        # 최종 출력 추출
        final_output = result_chunks[-1] if result_chunks else None
        if not final_output:
            return "생성 결과가 없습니다."
        
        # 생성된 결과 반환
        answer_text = "".join(piece.text for piece in final_output.outputs)
        logger.info(f"생성 완료: 길이={len(answer_text)}")
        
        return answer_text
        
    except Exception as e:
        logger.error(f"프롬프트 테스트 오류: {str(e)}", exc_info=True)
        return f"프롬프트 테스트 오류: {str(e)}"

@time_tracker
async def test_prompt_streaming(
    system_prompt: str,
    user_text: str,
    image_data: Optional[Union[str, bytes]],
    model: AsyncLLMEngine,
    tokenizer: Any,
    config: Any,
    request_id: str = None
):
    """
    프롬프트 테스트를 스트리밍 방식으로 수행합니다.
    
    Args:
        system_prompt: 테스트할 시스템 프롬프트
        user_text: 사용자 입력 텍스트
        image_data: 이미지 데이터 (base64 또는 바이너리)
        model: vLLM 엔진
        tokenizer: 토크나이저
        config: 설정
        request_id: 요청 ID
        
    Yields:
        str: 생성된 부분 텍스트
    """
    if not request_id:
        request_id = str(uuid.uuid4())
    
    logger.info(f"스트리밍 프롬프트 테스트 시작: request_id={request_id}")
    
    # 이미지 처리
    pil_image = None
    if image_data:
        try:
            pil_image = await prepare_image(image_data)
            logger.info(f"이미지 로드 성공: {pil_image.size if pil_image else None}")
        except Exception as e:
            logger.error(f"이미지 로드 실패: {str(e)}")
            yield f"이미지 로드 오류: {str(e)}"
            return
    
    try:
        # 멀티모달 메시지 설정
        if pil_image:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                config.model_id,
                use_fast=False
            )
            
            # 시스템 메시지와 사용자 메시지 구성
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": pil_image},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
            
            # 채팅 템플릿 적용
            prompt_string = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # 멀티모달 요청 구성
            generate_request = {
                "prompt": prompt_string,
                "multi_modal_data": {
                    "image": [pil_image]
                }
            }
        else:
            # 이미지 없는 경우, 일반 텍스트 프롬프트 구성
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id,
                use_fast=True,
                trust_remote_code=True
            )
            
            # 시스템 메시지와 사용자 메시지 구성
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
            
            # 채팅 템플릿 적용
            prompt_string = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 일반 텍스트 요청
            generate_request = prompt_string
        
        # 샘플링 파라미터 설정
        sampling_params = SamplingParams(
            max_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
            top_k=config.model.top_k,
            top_p=config.model.top_p,
            repetition_penalty=config.model.repetition_penalty,
        )
        
        # 부분 결과를 누적하기 위한 변수
        accumulated_text = ""
        
        # vLLM으로 스트리밍 생성
        async for output in model.generate(
            prompt=generate_request,
            sampling_params=sampling_params,
            request_id=request_id
        ):
            if not output.outputs:
                continue
                
            # 새 토큰 추출
            new_text = output.outputs[0].text
            if len(new_text) > len(accumulated_text):
                # 새로 추가된 부분만 yield
                new_chunk = new_text[len(accumulated_text):]
                accumulated_text = new_text
                yield new_chunk
        
    except Exception as e:
        logger.error(f"스트리밍 프롬프트 테스트 오류: {str(e)}", exc_info=True)
        yield f"오류 발생: {str(e)}"