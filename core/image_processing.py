# core/image_processing.py
"""
이미지 처리 관련 기능을 제공하는 모듈

주요 기능:
1. 이미지 로드 및 처리
2. 이미지에 대한 설명 생성
3. 멀티모달 요청 준비
"""

import base64
import io
import logging
import json
import re
import uuid
from typing import Dict, Any, Optional, Union
from PIL import Image

from utils.tracking import time_tracker
from prompt import IMAGE_DESCRIPTION_PROMPT

# vLLM 관련 임포트
from vllm.multimodal.utils import fetch_image

# 로깅 설정
logger = logging.getLogger("RAG.ImageProcessing")

@time_tracker
async def prepare_image(image_data: Union[str, bytes]) -> Optional[Image.Image]:
    """
    다양한 형식의 이미지 데이터를 PIL Image로 변환
    
    Args:
        image_data: 이미지 데이터 (URL 또는 base64 인코딩된 데이터)
        
    Returns:
        Optional[Image.Image]: 변환된 PIL 이미지, 실패 시 None
    """
    try:
        logger.info("이미지 변환 시작: 데이터 타입=%s", type(image_data))
        
        # URL인 경우
        if isinstance(image_data, str) and (image_data.startswith("http://") or image_data.startswith("https://")):
            logger.info("URL 형식 이미지 로드")
            pil_image = fetch_image(image_data)
            
        # Base64 인코딩 데이터인 경우
        else:
            logger.info("Base64 형식 이미지 로드")
            # Data URL 형식인 경우 헤더 제거
            if isinstance(image_data, str) and image_data.startswith("data:image/"):
                logger.info("'data:image/' 접두사 감지 - 분리")
                image_data = image_data.split(",", 1)[-1]
                
            # Base64 디코딩
            decoded = base64.b64decode(image_data)
            logger.info("Base64 디코딩 완료: %d 바이트", len(decoded))
            
            # PIL 이미지로 변환
            pil_image = Image.open(io.BytesIO(decoded)).convert("RGB")
            
        logger.info("이미지 로드 성공: 크기=%s", pil_image.size)
        return pil_image
        
    except Exception as e:
        logger.error("이미지 로드 실패: %s", str(e), exc_info=True)
        return None

@time_tracker
async def prepare_processor(model_id: str) -> Any:
    """
    이미지 처리를 위한 프로세서 로드
    
    Args:
        model_id: 모델 ID
        
    Returns:
        Any: 로드된 프로세서
    """
    try:
        from transformers import AutoProcessor
        
        logger.info(f"프로세서 로드: model_id='{model_id}'")
        processor = AutoProcessor.from_pretrained(
            model_id,
            use_fast=False  # 모델이 fast processor를 지원한다면 True로 시도 가능
        )
        
        logger.info("프로세서 로드 완료: 타입=%s", type(processor).__name__)
        return processor
        
    except Exception as e:
        logger.error("프로세서 로드 실패: %s", str(e), exc_info=True)
        raise

@time_tracker
async def prepare_multimodal_request(
    prompt: str, 
    pil_image: Image.Image, 
    model_id: Any, 
    tokenizer: Any
) -> Dict[str, Any]:
    """
    멀티모달 요청 준비
    
    Args:
        prompt: 텍스트 프롬프트
        pil_image: PIL 이미지
        model: 언어 모델
        tokenizer: 토크나이저
        
    Returns:
        Dict[str, Any]: 멀티모달 요청 객체
    """
    try:
        from transformers import AutoProcessor
        
        # 모델 프로세서 로드
        processor = await prepare_processor(model_id)
        
        logger.info("멀티모달 메시지 구성")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # 채팅 템플릿 적용
        prompt_string = processor.apply_chat_template(
            messages,
            tokenize=False,  # 토큰화 없이 문자열만 생성
            add_generation_prompt=True,
        )
        
        logger.info("멀티모달 요청 객체 생성: prompt_length=%d", len(prompt_string))
        generate_request = {
            "prompt": prompt_string,
            "multi_modal_data": {
                "image": [pil_image]  # 여러 이미지 지원 시 리스트에 추가
            }
        }
        
        return generate_request
        
    except Exception as e:
        logger.error("멀티모달 요청 준비 실패: %s", str(e), exc_info=True)
        raise

@time_tracker
async def image_query(http_query: Dict[str, Any], model, config) -> Dict[str, Any]:
    """
    이미지 입력을 받아 설명 생성 (비 스트리밍)
    
    Args:
        http_query: HTTP 요청 데이터
        model: 언어 모델
        config: 설정
        
    Returns:
        Dict[str, Any]: 이미지 설명 결과
    """
    logger.info("이미지 분석 쿼리 처리 시작")
    
    # 파라미터 파싱
    request_id = http_query.get("request_id", str(uuid.uuid4()))
    image_input = http_query.get("image_data")
    user_query = http_query.get("qry_contents", "이 이미지를 한국어로 잘 설명해주세요.")
    
    logger.info(f"요청 정보: request_id={request_id}, user_query='{user_query}'")
    
    # 이미지 변환
    pil_image = await prepare_image(image_input)
    if not pil_image:
        error_msg = "이미지 로드 실패"
        logger.error(error_msg)
        return {"type": "error", "message": error_msg}
    
    # 프로세서 로드
    try:
        processor = await prepare_processor(config.model_id)
    except Exception as e:
        error_msg = f"프로세서 로드 실패: {str(e)}"
        logger.error(error_msg)
        return {"type": "error", "message": error_msg}
    
    # 이미지 정보 및 프롬프트 구성
    image_info = "Image data provided"  # 실제 서비스에서는 이미지 메타데이터 추출 가능
    prompt_image_sorting = IMAGE_DESCRIPTION_PROMPT.format(image_info=image_info, user_query=user_query)
    
    # 메시지 구성
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt_image_sorting}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": pil_image},
                {"type": "text", "text": user_query},
            ],
        }
    ]
    
    # 채팅 템플릿 적용
    try:
        prompt_string = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        logger.info("프롬프트 생성 완료: length=%d", len(prompt_string))
    except Exception as e:
        error_msg = f"채팅 템플릿 적용 오류: {str(e)}"
        logger.error(error_msg)
        return {"type": "error", "message": error_msg}
    
    # 샘플링 파라미터 설정
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        max_tokens=config.model.max_new_tokens,
        temperature=config.model.temperature,
        top_k=config.model.top_k,
        top_p=config.model.top_p,
        repetition_penalty=config.model.repetition_penalty,
    )
    
    # vLLM으로 생성
    result_chunks = []
    try:
        # 멀티모달 요청 구성
        generate_request = {
            "prompt": prompt_string,
            "multi_modal_data": {
                "image": [pil_image]
            }
        }

        # 생성 실행
        async for out in model.generate(
            prompt=generate_request,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            result_chunks.append(out)
            
        logger.info("생성 완료: 청크 수=%d", len(result_chunks))

    except Exception as e:
        error_msg = f"생성 중 오류: {str(e)}"
        logger.error(error_msg)
        return {"type": "error", "message": error_msg}

    # 출력이 없는 경우
    if not result_chunks:
        logger.error("모델 출력 없음")
        return {"type": "error", "message": "모델에서 출력이 없습니다."}

    # 최종 출력 추출
    final_output = next((c for c in result_chunks if getattr(c, "finished", False)), result_chunks[-1])
    answer_text = "".join(piece.text for piece in final_output.outputs)
    logger.info(f"최종 답변: length={len(answer_text)}")
    
    # JSON 형식 추출 시도
    try:
        # JSON 블록 추출
        match = re.search(r'\{.*\}', answer_text, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
        else:
            raise ValueError("JSON 응답 형식 미확인")
            
        if "is_structured" not in result or "description" not in result:
            raise ValueError("응답에 필요한 키가 누락됨")
            
    except Exception as e:
        # 파싱 실패 시 기본 처리
        logger.warning(f"JSON 파싱 실패: {str(e)} - 기본값으로 대체")
        result = {"is_structured": False, "description": http_query.get("qry_contents", "")}
        
    return result

@time_tracker
async def image_streaming_query(
    http_query: Dict[str, Any], 
    model, 
    tokenizer, 
    config
) -> Any:
    """
    이미지 입력을 받아 스트리밍 방식으로 설명 생성
    
    Args:
        http_query: HTTP 요청 데이터
        model: 언어 모델
        tokenizer: 토크나이저
        config: 설정
        
    Yields:
        str: 생성된 부분 텍스트
    """
    logger.info("이미지 스트리밍 쿼리 처리 시작")
    
    # 파라미터 파싱
    request_id = http_query.get("request_id", str(uuid.uuid4()))
    image_input = http_query.get("image_data")
    user_query = http_query.get("qry_contents", "이 이미지를 설명해주세요.")
    
    logger.info(f"요청 정보: request_id={request_id}, user_query='{user_query}'")
    
    # 이미지 변환
    try:
        pil_image = await prepare_image(image_input)
        if not pil_image:
            error_msg = "이미지 로드 실패"
            logger.error(error_msg)
            yield {"type": "error", "message": error_msg}
            return
    except Exception as e:
        error_msg = f"이미지 로드 실패: {str(e)}"
        logger.error(error_msg)
        yield {"type": "error", "message": error_msg}
        return
    
    # 프로세서 로드
    try:
        processor = await prepare_processor(config.model_id)
    except Exception as e:
        error_msg = f"프로세서 로드 실패: {str(e)}"
        logger.error(error_msg)
        yield {"type": "error", "message": error_msg}
        return
    
    # 메시지 구성
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": pil_image},
                {"type": "text", "text": user_query},
            ],
        }
    ]
    
    # 채팅 템플릿 적용
    try:
        prompt_string = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        logger.info("프롬프트 생성 완료")
    except Exception as e:
        error_msg = f"채팅 템플릿 적용 오류: {str(e)}"
        logger.error(error_msg)
        yield {"type": "error", "message": error_msg}
        return
    
    # 샘플링 파라미터 설정
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        max_tokens=config.model.max_new_tokens,
        temperature=config.model.temperature,
        top_k=config.model.top_k,
        top_p=config.model.top_p,
        repetition_penalty=config.model.repetition_penalty,
    )
    
    # 멀티모달 요청 구성
    generate_request = {
        "prompt": prompt_string,
        "multi_modal_data": {
            "image": [pil_image]
        }
    }
    
    # 스트리밍 생성
    from core.generation import collect_vllm_text_stream
    
    async for partial_chunk in collect_vllm_text_stream(
        generate_request, model, sampling_params, request_id
    ):
        yield partial_chunk