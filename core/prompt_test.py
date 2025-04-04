# prompt_testing.py (수정된 부분만)
import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, Union, List
from PIL import Image

# 유틸리티 임포트
from utils.tracking import time_tracker
from core.image_processing import prepare_image
from core.pdf_processor import pdf_to_prompt_context  # 새로 추가

# vLLM 관련 임포트
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine

# 로깅 설정
logger = logging.getLogger("PROMPT_TESTING")

@time_tracker
async def test_prompt_with_image(
    system_prompt: str,
    user_text: str,
    file_data: Optional[Union[str, bytes]] = None,
    file_type: Optional[str] = None,
    model: AsyncLLMEngine = None,
    tokenizer: Any = None,
    config: Any = None,
    request_id: str = None
) -> str:
    """
    파일(이미지 또는 PDF)과 함께 프롬프트를 테스트합니다.
    
    Args:
        system_prompt: 테스트할 시스템 프롬프트
        user_text: 사용자 입력 텍스트
        file_data: 파일 데이터 (base64 또는 바이너리)
        file_type: 파일 타입 ('image' 또는 'pdf')
        model: vLLM 엔진
        tokenizer: 토크나이저
        config: 설정
        request_id: 요청 ID
        
    Returns:
        str: 생성된 전체 텍스트
    """
    if not request_id:
        request_id = str(uuid.uuid4())
    
    logger.info(f"프롬프트 테스트 시작: request_id={request_id}, file_type={file_type}")
    
    # 파일 처리
    pil_image = None
    pil_images= []
    pdf_context = {}
    pdf_images = []
    
    if file_data:
        if file_type == 'image':
            try:
                pil_image = await prepare_image(file_data)
                logger.info(f"이미지 로드 성공: {pil_image.size if pil_image else None}")
            except Exception as e:
                logger.error(f"이미지 로드 실패: {str(e)}")
                return f"이미지 로드 오류: {str(e)}"
        elif file_type == 'pdf':
            try:
                pdf_context = await pdf_to_prompt_context(file_data)
                if "error" in pdf_context:
                    logger.error(f"PDF 처리 실패: {pdf_context['error']}")
                    return f"PDF 처리 오류: {pdf_context['error']}"
                    
                # PDF에서 추출한 텍스트를 사용자 텍스트에 추가
                user_text = f"{user_text}\n\nPDF 내용:\n{pdf_context['text_context']}"
                
                # 추출한 모든 이미지 처리
                pdf_images = pdf_context.get("images", [])
                if pdf_images:
                    import base64
                    import io
                    
                    # 모든 이미지를 PIL 이미지로 변환하여 리스트에 추가
                    for img_data in pdf_images:
                        try:
                            img_bytes = base64.b64decode(img_data["base64"])
                            img = Image.open(io.BytesIO(img_bytes))
                            pil_images.append(img)
                            logger.info(f"PDF 이미지 추가: 페이지 {img_data['page']}, ID {img_data['image_id']}, 크기 {img.size}")
                        except Exception as e:
                            logger.warning(f"이미지 변환 오류 (무시됨): {str(e)}")
                            
                    logger.info(f"PDF에서 추출한 전체 이미지 수: {len(pil_images)}")
                    
            except Exception as e:
                logger.error(f"PDF 처리 중 오류: {str(e)}")
                return f"PDF 처리 오류: {str(e)}"
    
    try:
        # 멀티모달 메시지 설정 (이미지가 있는 경우)
        if pil_images:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                config.model_id,
                use_fast=False
            )
            
            # 시스템 메시지와 사용자 메시지 구성
            # 복수의 이미지를 포함하도록 수정
            message_content = [{"type": "text", "text": user_text}]
            
            # 이미지 추가 (최대 5개로 제한해 모델이 처리할 수 있는 범위 내에서 유지)
            max_images_for_model = min(5, len(pil_images))
            for i in range(max_images_for_model):
                message_content.insert(0, {"type": "image", "url": pil_images[i]})
                
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": message_content
                }
            ]
            
            # 채팅 템플릿 적용
            prompt_string = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # 멀티모달 요청 구성 (모든 이미지 포함)
            generate_request = {
                "prompt": prompt_string,
                "multi_modal_data": {
                    "image": pil_images[:max_images_for_model]  # 최대 5개 이미지로 제한
                }
            }
            
            if len(pil_images) > max_images_for_model:
                logger.warning(f"이미지가 너무 많아 첫 {max_images_for_model}개만 사용합니다 (총 {len(pil_images)}개 발견)")
                
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
        
        # PDF에서 추출한 이미지 정보도 반환 (JSON으로 변환)
        if pdf_images and len(pdf_images) > 1:
            answer_text += f"\n\n[PDF에서 총 {len(pdf_images)}개 이미지를 추출했습니다. 첫 번째 이미지만 사용되었습니다.]"
        
        return answer_text
        
    except Exception as e:
        logger.error(f"프롬프트 테스트 오류: {str(e)}", exc_info=True)
        return f"프롬프트 테스트 오류: {str(e)}"

@time_tracker
async def test_prompt_streaming(
    system_prompt: str,
    user_text: str,
    file_data: Optional[Union[str, bytes]] = None,
    file_type: Optional[str] = None,
    model: AsyncLLMEngine = None,
    tokenizer: Any = None,
    config: Any = None,
    request_id: str = None
):
    """
    프롬프트 테스트를 스트리밍 방식으로 수행합니다.
    
    Args:
        system_prompt: 테스트할 시스템 프롬프트
        user_text: 사용자 입력 텍스트
        file_data: 파일 데이터 (base64 또는 바이너리)
        file_type: 파일 타입 ('image' 또는 'pdf')
        model: vLLM 엔진
        tokenizer: 토크나이저
        config: 설정
        request_id: 요청 ID
        
    Yields:
        str: 생성된 부분 텍스트
    """
    if not request_id:
        request_id = str(uuid.uuid4())
    
    logger.info(f"스트리밍 프롬프트 테스트 시작: request_id={request_id}, file_type={file_type}")
    
    # 파일 처리
    pil_image = None
    pil_images = []  # 단일 이미지 대신 이미지 리스트로 변경
    pdf_context = {}
    pdf_images = []
    
    if file_data:
        if file_type == 'image':
            try:
                pil_image = await prepare_image(file_data)
                logger.info(f"이미지 로드 성공: {pil_image.size if pil_image else None}")
            except Exception as e:
                logger.error(f"이미지 로드 실패: {str(e)}")
                yield f"이미지 로드 오류: {str(e)}"
                return
        elif file_type == 'pdf':
            try:
                yield "PDF 분석 중..."
                pdf_context = await pdf_to_prompt_context(file_data, max_pages=10, max_images=10)
                if "error" in pdf_context:
                    logger.error(f"PDF 처리 실패: {pdf_context['error']}")
                    yield f"PDF 처리 오류: {pdf_context['error']}"
                    return
                
                # PDF에서 추출한 텍스트를 사용자 텍스트에 추가
                user_text = f"{user_text}\n\nPDF 내용:\n{pdf_context['text_context']}"
                
                # 추출된 이미지 처리
                pdf_images = pdf_context.get("images", [])
                if pdf_images:
                    import base64
                    import io
                    
                    # 모든 이미지를 PIL 이미지로 변환하여 리스트에 추가
                    for img_data in pdf_images:
                        try:
                            img_bytes = base64.b64decode(img_data["base64"])
                            img = Image.open(io.BytesIO(img_bytes))
                            pil_images.append(img)
                        except Exception as e:
                            logger.warning(f"이미지 변환 오류 (무시됨): {str(e)}")
                    
                    max_images_for_model = min(5, len(pil_images))
                    yield f"PDF에서 {len(pdf_images)}개 이미지 추출 완료. {max_images_for_model}개 이미지를 사용합니다.\n"
                    
                    if len(pil_images) > max_images_for_model:
                        yield f"(참고: 모델 제한으로 인해 {len(pil_images)}개 중 {max_images_for_model}개만 사용됩니다)\n"
                else:
                    yield "PDF 분석 완료. 이미지가 없습니다.\n"
                    
            except Exception as e:
                logger.error(f"PDF 처리 중 오류: {str(e)}")
                yield f"PDF 처리 오류: {str(e)}"
                return
    
    try:
        # 멀티모달 메시지 설정 (이미지가 있는 경우)
        if pil_images:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                config.model_id,
                use_fast=False
            )
            
            # 시스템 메시지와 사용자 메시지 구성
            # 복수의 이미지를 포함하도록 수정
            message_content = [{"type": "text", "text": user_text}]
            
            # 이미지 추가 (최대 5개로 제한해 모델이 처리할 수 있는 범위 내에서 유지)
            max_images_for_model = min(5, len(pil_images))
            for i in range(max_images_for_model):
                message_content.insert(0, {"type": "image", "url": pil_images[i]})
                
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": message_content
                }
            ]
            
            # 채팅 템플릿 적용
            prompt_string = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # 멀티모달 요청 구성 (모든 이미지 포함)
            generate_request = {
                "prompt": prompt_string,
                "multi_modal_data": {
                    "image": pil_images[:max_images_for_model]  # 최대 5개 이미지로 제한
                }
            }
            
            if len(pil_images) > max_images_for_model:
                logger.warning(f"이미지가 너무 많아 첫 {max_images_for_model}개만 사용합니다 (총 {len(pil_images)}개 발견)")
                
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
        
        # 가장 중요한 부분: 실제 스트리밍 로직
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