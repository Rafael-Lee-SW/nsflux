"""
PDF 처리 모듈 - Fitz(PyMuPDF)를 사용하여 PDF에서 텍스트, 표, 이미지를 추출합니다.
"""
import io
import base64
import json
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger("PDF_PROCESSOR")

async def process_pdf(pdf_content, max_pages=None):
    """
    PDF 파일의 내용을 처리하여 텍스트, 표, 이미지를 추출합니다.
    
    Args:
        pdf_content: PDF 파일의 바이너리 콘텐츠 또는 base64 문자열
        max_pages: 처리할 최대 페이지 수 (None이면 모든 페이지)
        
    Returns:
        dict: PDF 콘텐츠(텍스트, 표, 이미지, 메타데이터 등)를 포함하는 딕셔너리
    """
    logger.info("PDF 처리 시작")
    
    # base64 문자열인 경우 디코딩
    if isinstance(pdf_content, str):
        try:
            pdf_content = base64.b64decode(pdf_content)
        except Exception as e:
            logger.error(f"Base64 디코딩 실패: {e}")
            return {"error": f"Base64 디코딩 실패: {e}"}
    
    try:
        # PDF 문서 열기
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        num_pages = doc.page_count
        
        if max_pages is not None:
            num_pages = min(num_pages, max_pages)
        
        # 결과를 저장할 딕셔너리
        result = {
            "metadata": {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "total_pages": doc.page_count,
                "processed_pages": num_pages
            },
            "pages": []
        }
        
        # 각 페이지 처리
        for page_num in range(num_pages):
            page = doc[page_num]
            page_data = {
                "page_number": page_num + 1,
                "text": page.get_text(),
                "tables": extract_tables(page),
                "images": extract_images(page)
            }
            result["pages"].append(page_data)
            
        logger.info(f"PDF 처리 완료: {num_pages}페이지, 이미지 {sum(len(page['images']) for page in result['pages'])}개, 표 {sum(len(page['tables']) for page in result['pages'])}개")
        
        return result
    
    except Exception as e:
        logger.error(f"PDF 처리 중 오류: {e}")
        return {"error": f"PDF 처리 실패: {e}"}
    
def extract_tables(page):
    """
    PDF 페이지에서 표 추출 시도
    """
    tables = []
    
    # 테이블 감지 시도
    try:
        # PyMuPDF의 내장 테이블 감지 기능 사용
        tab = page.find_tables()
        if tab.tables:
            for idx, table in enumerate(tab.tables):
                rows = []
                for cells in table.extract():
                    rows.append([cell.text for cell in cells])
                tables.append({
                    "table_id": idx,
                    "rows": rows,
                    "bbox": list(table.bbox)  # 테이블 위치 (x0, y0, x1, y1)
                })
    except Exception as e:
        logging.warning(f"테이블 추출 중 오류: {e}")
    
    return tables

def extract_images(page):
    """
    PDF 페이지에서 이미지 추출
    """
    images = []
    
    # 이미지 목록 가져오기
    img_list = page.get_images(full=True)
    
    for img_idx, img_info in enumerate(img_list):
        try:
            xref = img_info[0]  # 이미지 참조 번호
            base_image = page.parent.extract_image(xref)
            
            if not base_image:
                continue
                
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # PIL을 사용하여 이미지 열기
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # 이미지를 base64로 변환
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # 이미지 정보 저장
            images.append({
                "image_id": img_idx,
                "format": image_ext,
                "width": pil_image.width,
                "height": pil_image.height,
                "base64": img_base64
            })
            
        except Exception as e:
            logging.warning(f"이미지 {img_idx} 추출 중 오류: {e}")
    
    return images

async def pdf_to_prompt_context(pdf_content, max_pages=10, max_images=10):
    """
    PDF를 처리하여 LLM 프롬프트에 적합한 컨텍스트 문자열로 변환합니다.
    
    Args:
        pdf_content: PDF 바이너리 또는 base64 문자열
        max_pages: 처리할 최대 페이지 수
        max_images: 포함할 최대 이미지 수
        
    Returns:
        dict: 추출된 텍스트, 이미지 목록을 포함하는 딕셔너리
    """
    # PDF 처리
    pdf_data = await process_pdf(pdf_content, max_pages)
    
    if "error" in pdf_data:
        return {"error": pdf_data["error"]}
    
    # 텍스트 컨텍스트 구성
    context = f"PDF 문서 분석:\n\n"
    context += f"제목: {pdf_data['metadata']['title']}\n"
    context += f"저자: {pdf_data['metadata']['author']}\n"
    context += f"총 페이지: {pdf_data['metadata']['total_pages']} (처리된 페이지: {pdf_data['metadata']['processed_pages']})\n\n"
    
    # 각 페이지 내용 추가
    all_images = []
    
    for page in pdf_data["pages"]:
        context += f"--- 페이지 {page['page_number']} ---\n\n"
        context += f"{page['text']}\n\n"
        
        # 테이블 정보 추가
        if page["tables"]:
            for table_idx, table in enumerate(page["tables"]):
                context += f"[표 {page['page_number']}-{table_idx+1}]\n"
                for row in table["rows"]:
                    context += " | ".join(row) + "\n"
                context += "\n"
        
        # 이미지 정보 모으기 (별도로 반환하기 위해)
        for img in page["images"]:
            if len(all_images) < max_images:  # 최대 이미지 수 제한
                all_images.append({
                    "page": page["page_number"],
                    "image_id": img["image_id"],
                    "format": img["format"],
                    "width": img["width"],
                    "height": img["height"],
                    "base64": img["base64"]
                })
    
    return {
        "text_context": context,
        "images": all_images
    }