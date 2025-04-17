import json
import numpy as np
import torch
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import logging
from loguru import logger
import os
import time
from functools import wraps
from transformers import AutoModel, AutoTokenizer

# 로깅 설정
logger.remove()
logger.add(
    lambda msg: print(msg),
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# 성능 측정을 위한 데코레이터
def time_tracker(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} 실행 시간: {end_time - start_time:.4f}초")
        return result
    return wrapper

# 랜덤 시드 설정
@time_tracker
def random_seed(seed):
    """랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 데이터 로드 함수
@time_tracker
def load_data(data_path, image_base_path):
    """벡터 데이터베이스 로드"""
    logger.info(f"데이터 로드 시작: {data_path}")
    
    try:
        with open(data_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        
        # 데이터 전처리
        file_names = []
        chunk_ids = []
        titles = []
        times = []
        vectors = []
        texts = []
        texts_short = []
        texts_vis = []
        file_path = []
        missing_time = 0
        
        for file_obj in data:
            for chunk in file_obj["chunks"]:
                file_names.append(file_obj["file_name"])
                chunk_ids.append(chunk.get("chunk_id", 0))
                try:
                    arr = np.array(chunk["vector"])
                    vectors.append(arr)
                except Exception as e:
                    logger.warning(f"벡터 변환 오류: {e} → 빈 벡터로 대체")
                    vectors.append(np.zeros((1, 768), dtype=np.float32))
                
                titles.append(chunk["title"])
                
                # 날짜 파싱
                if chunk["date"]:
                    try:
                        times.append(datetime.strptime(chunk["date"], "%Y-%m-%d"))
                    except ValueError:
                        logger.warning(f"잘못된 날짜 형식: {chunk['date']} → 기본 날짜로 대체")
                        times.append(datetime.strptime("2023-10-31", "%Y-%m-%d"))
                        missing_time += 1
                else:
                    missing_time += 1
                    times.append(datetime.strptime("2023-10-31", "%Y-%m-%d"))
                
                texts.append(chunk["text"])
                texts_short.append(chunk["text_short"])
                texts_vis.append(chunk["text_vis"])
                
                # 파일 이미지 경로 추가
                if chunk["file_path"] is not None:
                    file_path.append(os.path.join(image_base_path, chunk["file_path"]))
                else:
                    file_path.append(None) # 이미지가 없는 파일은 None으로 저장됨.
        
        # 실제 텐서로 변환
        try:
            vectors = np.array(vectors)
            vectors = torch.from_numpy(vectors).to(torch.float32)
        except Exception as e:
            logger.error(f"최종 벡터 텐서 변환 오류: {str(e)}")
        
        processed_data = {
            "file_names": file_names,
            "chunk_ids": chunk_ids,
            "titles": titles,
            "times": times,
            "vectors": vectors,
            "texts": texts,
            "texts_short": texts_short,
            "texts_vis": texts_vis,
            "file_path": file_path,
        }
        
        logger.info(f"데이터 로드 완료: {len(titles)}개 문서, 누락된 날짜: {missing_time}개")
        return processed_data
        
    except Exception as e:
        logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return None

# 벡터 형식 디버깅
def debug_vector_format(data):
    """벡터 형식 디버깅"""
    logger.debug("===== 벡터 형식 검사 시작 =====")
    for f_i, file_obj in enumerate(data):
        file_name = file_obj.get("file_name", f"Unknown_{f_i}")
        chunks = file_obj.get("chunks", [])
        for c_i, chunk in enumerate(chunks):
            vector_data = chunk.get("vector", None)
            if vector_data is None:
                continue
            try:
                arr = np.array(vector_data)
                shape = arr.shape
                logger.debug(f"file={file_name}, chunk_index={c_i} → shape={shape}")
            except Exception as e:
                logger.debug(f"file={file_name}, chunk_index={c_i} → vector 변환 실패: {str(e)}")
    logger.debug("===== 벡터 형식 검사 종료 =====")

def vectorize_content(text: str) -> List[float]:
    """
    텍스트를 벡터로 변환합니다.
    """
    try:
        # 임베딩 모델 로드
        model = AutoModel.from_pretrained("BM-K/KoSimCSE-roberta-multitask")
        tokenizer = AutoTokenizer.from_pretrained("BM-K/KoSimCSE-roberta-multitask")
        model.eval()
        
        # 텍스트 토큰화 및 벡터화
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"벡터화 중 오류 발생: {str(e)}")
        return [0.0] * 768  # 기본 벡터 반환

def normalize_text_vis(text: str) -> str:
    """
    텍스트를 시각화용으로 정규화합니다.
    """
    # HTML 특수문자 이스케이프
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # 줄바꿈 처리
    text = text.replace("\n", "<br>")
    
    # 공백 처리
    text = " ".join(text.split())
    
    return text 