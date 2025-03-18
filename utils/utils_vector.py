# utils/utils_vector.py
import json
import numpy as np
import torch
import random
import shutil
from datetime import datetime, timedelta
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
)

# Define the minimum valid file size (e.g., 10MB)
MIN_WEIGHT_SIZE = 10 * 1024 * 1024

# For tracking execution time of functions
from utils.tracking import time_tracker

# Logging
import logging
logging.basicConfig(level=logging.DEBUG)

# ---------------------- 벡터화 -----------------------
import yaml
from box import Box
# Configuration
with open("./config.yaml", "r") as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)

# 임베딩 모델 및 토크나이저 (청크 벡터화를 위해 별도 로드)
embedding_model = AutoModel.from_pretrained(config.embed_model_id, cache_dir=config.cache_dir)
embedding_tokenizer = AutoTokenizer.from_pretrained(config.embed_model_id, cache_dir=config.cache_dir)
embedding_model.eval()

# -------------------- 벡터화 함수 --------------------
@time_tracker
def vectorize_content(content):
    try:
        inputs = embedding_tokenizer(content, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = embedding_model(**inputs, return_dict=False)
        # 첫 토큰의 임베딩을 사용 (1D 벡터)
        vector = outputs[0][:, 0, :].squeeze(0).tolist()
        
        # 벡터 일관성 확인
        expected_dim = 768  # 임베딩 모델 차원에 맞게 조정
        
        # 리스트가 아닌 경우 변환 시도
        if not isinstance(vector, list):
            print(f"경고: 벡터가 리스트가 아님, 타입: {type(vector)}")
            try:
                vector = list(vector)
            except Exception as e:
                print("오류: 벡터를 리스트로 변환 실패:", e)
                vector = [0.0] * expected_dim  # 기본 벡터 제공
        
        # 벡터 차원 확인 및 조정
        if len(vector) != expected_dim:
            print(f"경고: 벡터 차원 불일치. 예상: {expected_dim}, 실제: {len(vector)}")
            if len(vector) < expected_dim:
                # 부족한 차원은 0으로 패딩
                vector.extend([0.0] * (expected_dim - len(vector)))
            else:
                # 초과 차원은 자르기
                vector = vector[:expected_dim]
        
        # 기존 파일 형식과 일치하도록 항상 2차원 배열 형식으로 반환 ([[...] 형태])
        if vector and not isinstance(vector[0], list):
            return [vector]
        return vector
    except Exception as e:
        print(f"vectorize_content 함수 오류: {str(e)}")
        # 오류 시 기본 벡터 반환 (2차원 형식)
        return [[0.0] * 768]

# -------------------- 텍스트 출력 필드 정규화 함수 --------------------
def normalize_text_vis(text_vis):
    """
    text_vis가 이미 올바른 리스트-딕셔너리 구조이면 그대로 반환하고,
    그렇지 않은 경우 기본 구조로 감싸서 반환합니다.
    """
    if isinstance(text_vis, list) and len(text_vis) > 0 and isinstance(text_vis[0], dict):
        # 필요한 키가 존재하는지 확인
        if all(k in text_vis[0] for k in ("rsp_type", "rsp_tit", "rsp_data")):
            return text_vis
    if isinstance(text_vis, str):
        return [{
            "rsp_type": "TT",
            "rsp_tit": "",
            "rsp_data": text_vis
        }]
    return [{
        "rsp_type": "TT",
        "rsp_tit": "",
        "rsp_data": str(text_vis)
    }]

# -------------------- 데이터셋 진단 및 수정 도구 --------------------
# 데이터셋 진단 및 복구 함수 (utils.py 또는 별도 파일에 추가)
def diagnose_and_fix_dataset(data_path, output_path=None):
    """
    데이터셋의 벡터 차원 문제를 진단하고 수정합니다.
    """
    try:
        print(f"데이터셋 진단 중: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"데이터셋 내 파일 수: {len(data)}")
        dimensions = {}
        fixed_count = 0
        problem_count = 0
        
        # 1단계: 가장 흔한 차원 찾기
        for file_idx, file in enumerate(data):
            file_name = file.get("file_name", f"Unknown-{file_idx}")
            for chunk_idx, chunk in enumerate(file.get("chunks", [])):
                if "vector" in chunk and chunk["vector"]:
                    vector = chunk["vector"]
                    try:
                        if isinstance(vector, list):
                            dim = len(vector)
                            dimensions[dim] = dimensions.get(dim, 0) + 1
                        else:
                            print(f"벡터가 리스트가 아님: {file_name}, 청크 {chunk_idx}")
                            problem_count += 1
                    except Exception as e:
                        print(f"벡터 길이 확인 실패: {file_name}, 청크 {chunk_idx} - {str(e)}")
                        problem_count += 1
        
        if dimensions:
            # 가장 흔한 차원 찾기
            expected_dim = max(dimensions.items(), key=lambda x: x[1])[0]
            print(f"가장 흔한 벡터 차원: {expected_dim} (총 {dimensions[expected_dim]}개 발견)")
            print(f"발견된 모든 차원: {dimensions}")
        else:
            print("데이터셋에서 유효한 벡터를 찾을 수 없습니다!")
            return False
        
        # 2단계: 잘못된 차원의 벡터 수정
        for file_idx, file in enumerate(data):
            file_name = file.get("file_name", f"Unknown-{file_idx}")
            for chunk_idx, chunk in enumerate(file.get("chunks", [])):
                if "vector" in chunk and chunk["vector"]:
                    vector = chunk["vector"]
                    try:
                        if not isinstance(vector, list):
                            print(f"리스트가 아닌 벡터 수정 시도: {file_name}, 청크 {chunk_idx}")
                            try:
                                vector = list(vector)
                                chunk["vector"] = vector
                                fixed_count += 1
                            except:
                                # 변환 실패 시 빈 벡터 생성
                                chunk["vector"] = [0.0] * expected_dim
                                fixed_count += 1
                                print(f"리스트 변환 실패, 기본 벡터 사용")
                        
                        dim = len(vector)
                        if dim != expected_dim:
                            print(f"벡터 차원 수정: {file_name}, 청크 {chunk_idx} (차원: {dim})")
                            if dim < expected_dim:
                                # 0으로 패딩
                                chunk["vector"] = vector + [0.0] * (expected_dim - dim)
                            else:
                                # 자르기
                                chunk["vector"] = vector[:expected_dim]
                            fixed_count += 1
                    except Exception as e:
                        print(f"벡터 처리 중 오류: {file_name}, 청크 {chunk_idx} - {str(e)}")
                        problem_count += 1
        
        print(f"고정된 벡터 수: {fixed_count}, 문제 벡터 수: {problem_count}")
        
        # 수정된 데이터셋 저장
        if output_path is None:
            output_path = data_path
        
        # 덮어쓰기 전 백업 생성
        if output_path == data_path:
            backup_path = f"{data_path}.bak"
            print(f"백업 생성: {backup_path}")
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"수정된 데이터셋 저장: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True
    
    except Exception as e:
        print(f"데이터셋 진단 중 오류: {str(e)}")
        return False
    