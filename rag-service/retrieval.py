import numpy as np
import torch
import torch.nn.functional as F
import rank_bm25
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from loguru import logger
from utils import time_tracker
import asyncio
import ray

@time_tracker
async def retrieve(
    query: str, 
    data: Dict[str, Any], 
    top_n: int, 
    embed_model, 
    embed_tokenizer
) -> Tuple[str, List[Dict]]:
    """
    주어진 쿼리와 관련된 문서를 검색하고 순위를 매긴 뒤 반환
    
    Args:
        query: 검색 쿼리 또는 키워드
        data: 벡터화된 문서 데이터베이스
        top_n: 반환할 상위 문서 수
        embed_model: 임베딩 모델
        embed_tokenizer: 임베딩 토크나이저
    
    Returns:
        Tuple[str, List[Dict]]: (문서 내용 문자열, 문서 메타데이터 리스트)
    """
    logger.info(f"검색 시작: 쿼리='{query}', 문서 수={len(data['vectors'])}")
    
    try:
        # 유사도 스코어 계산
        sim_score = await cal_sim_score(query, data["vectors"], embed_model, embed_tokenizer)
        logger.info(f"유사도 스코어 계산 완료: shape={sim_score.shape}")
        
        # BM25 스코어 계산
        # bm25_score = cal_bm25_score(query, data["texts_short"], embed_tokenizer)
        # logger.info(f"BM25 스코어 계산 완료: shape={bm25_score.shape}")
        
        # 스코어 정규화
        scaled_sim_score = min_max_scaling(sim_score)
        # scaled_bm25_score = min_max_scaling(bm25_score)
        
        # 최종 스코어 계산 (가중치 적용)
        score = scaled_sim_score
        # score = scaled_sim_score * 0.7 + scaled_bm25_score * 0.3
        score_values = score[:, 0, 0]
        
        # 상위 N개 문서 선택
        top_k = score[:, 0, 0].argsort()[-top_n:][::-1]
        
        logger.info(f"상위 {top_n}개 문서 인덱스: {top_k}")
        logger.info(f"상위 {top_n}개 문서 스코어: {[score[:, 0, 0][i] for i in top_k]}")
        
        # 문서 컨텐츠와 메타데이터 구성
        documents = ""
        documents_list = []
        
        for i, index in enumerate(top_k):
            score_str = f"{score_values[index]:.4f}"
            # 문서 텍스트 컨텐츠 구성
            documents += f"{i+1}번째 검색자료 (출처:{data['file_names'][index]}) :\n{data['texts'][index]}, , Score: {score_str}\n"
            
            # 문서 메타데이터 구성
            documents_list.append({
                "file_name": data["file_names"][index],
                "title": data["titles"][index],
                "contents": data["texts_vis"][index],
                "chunk_id": data["chunk_ids"][index],
                "file_path": data["file_path"][index],
                "text_short": data["texts_short"][index]
            })
            logger.info(data["file_path"][index])
        
        logger.info("검색 완료: %d개 문서 반환", len(documents_list))
        return documents, documents_list
        
    except Exception as e:
        logger.error(f"검색 중 오류 발생: {str(e)}", exc_info=True)
        return "", []

@time_tracker
async def cal_sim_score(
    query: str, 
    chunks: Union[torch.Tensor, List[torch.Tensor]], 
    embed_model, 
    embed_tokenizer
) -> np.ndarray:
    """
    쿼리와 문서 청크 간의 코사인 유사도 스코어 계산
    
    Args:
        query: 검색 쿼리
        chunks: 문서 청크 임베딩 벡터 (텐서 또는 텐서 리스트)
        embed_model: 임베딩 모델
        embed_tokenizer: 임베딩 토크나이저
        
    Returns:
        np.ndarray: 유사도 스코어 배열
    """
    logger.info(f"유사도 계산 시작: 쿼리='{query}'")
    
    # 쿼리 임베딩 - 비동기 방식 사용
    if hasattr(embed_model, 'embed_async'):
        query_vector = await embed_model.embed_async(query)
    else:
        query_vector = embed(query, embed_model, embed_tokenizer)
    
    if len(query_vector.shape) == 1:
        query_vector = query_vector.unsqueeze(0)
    
    # 각 청크와의 유사도 계산
    passage_emb = F.normalize(chunks, p=2, dim=-1) # (Batch, 1, Dim)
    query_emb   = F.normalize(query_vector, p=2, dim=-1) # (1, Dim)

    score = (passage_emb @ query_emb.T) # (Batch, 1, 1)
    
    return np.array(score)

@time_tracker
def cal_bm25_score(
    query: str, 
    indexes: List[str], 
    embed_tokenizer
) -> np.ndarray:
    """
    BM25 알고리즘을 사용하여 쿼리와 문서 간의 관련성 스코어 계산
    
    Args:
        query: 검색 쿼리
        indexes: 문서 텍스트 목록
        embed_tokenizer: 토크나이저
        
    Returns:
        np.ndarray: BM25 스코어 배열
    """
    logger.info(f"BM25 계산 시작: 쿼리='{query}', 문서 수={len(indexes)}")
    
    if not indexes:
        logger.warning("BM25에 빈 문서 리스트가 제공됨")
        return np.zeros(0)
    
    # 토큰화
    tokenized_corpus = []
    for i, text in enumerate(indexes):
        try:
            tokens = embed_tokenizer(
                text,
                return_token_type_ids=False,
                return_attention_mask=False,
                return_offsets_mapping=False,
            )
            tokens = embed_tokenizer.convert_ids_to_tokens(tokens["input_ids"])
            
            if len(tokens) == 0:
                logger.warning(f"문서 {i}가 빈 토큰 리스트로 토큰화됨")
                tokens = ["<empty>"]
                
            tokenized_corpus.append(tokens)
            
        except Exception as e:
            logger.error(f"문서 {i} 토큰화 실패: {str(e)}")
            tokenized_corpus.append(["<error>"])
    
    # BM25 계산
    try:
        bm25 = rank_bm25.BM25Okapi(tokenized_corpus)
        tokenized_query = embed_tokenizer.convert_ids_to_tokens(embed_tokenizer(query)["input_ids"])
        scores = bm25.get_scores(tokenized_query)
        
        # NaN/Inf 값 처리
        if np.isnan(scores).any() or np.isinf(scores).any():
            logger.warning("BM25에서 NaN/Inf 스코어 발생 - 0으로 대체")
            scores = np.nan_to_num(scores)
        
        logger.info(
            f"BM25 스코어: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}"
        )
        return scores
        
    except Exception as e:
        logger.error(f"BM25 계산 실패: {str(e)}")
        return np.zeros(len(indexes))

@time_tracker
def embed(query: str, embed_model, embed_tokenizer) -> torch.Tensor:
    """
    텍스트를 임베딩 벡터로 변환

    Args:
        query: 임베딩할 텍스트
        embed_model: 임베딩 모델
        embed_tokenizer: 임베딩 토크나이저
    
    Returns:
        torch.Tensor: 임베딩 벡터
    """
    logger.info(f"임베딩 시작: '{query}'")
    with torch.no_grad():
        inputs = embed_tokenizer(query, max_length=4096, padding="max_length", truncation=True, return_tensors="pt").to(embed_model.device)
        outputs = embed_model(**inputs)
        # Last Sequence Token is used as Embedding V
        embeddings = outputs.last_hidden_state[:,-1].cpu() # (Batch, Embed Dims) == (1,4096)

    logger.info("임베딩 완료")
    return embeddings

@time_tracker
def min_max_scaling(arr: np.ndarray) -> np.ndarray:
    """
    배열을 0-1 범위로 정규화
    
    Args:
        arr: 정규화할 배열
        
    Returns:
        np.ndarray: 정규화된 배열
    """
    arr_min = arr.min()
    arr_max = arr.max()
    
    if arr_max == arr_min:
        logger.warning("정규화 범위가 0 - 0으로 들어옴")
        return np.zeros_like(arr)
        
    return (arr - arr_min) / (arr_max - arr_min)

@time_tracker
def sort_by_time(time_bound: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    시간 범위에 따라 데이터 필터링
    
    Args:
        time_bound: 시간 범위 ("all" 또는 "시작일:종료일" 형식)
        data: 원본 데이터
        
    Returns:
        Dict[str, Any]: 필터링된 데이터
    """
    original_count = len(data["times"])
    logger.info(f"시간 필터링 전 문서 수: {original_count}")
    
    # 전체 기간 선택 시 원본 반환
    if time_bound == "all" or time_bound == "1900-01-01:2099-01-01":
        logger.info("전체 기간 사용 - 모든 문서 포함")
        return data
    
    # 시간 범위 파싱
    date_format = "%Y-%m-%d"
    target_date_start = datetime.strptime(time_bound.split(":")[0], date_format)
    target_date_end = datetime.strptime(time_bound.split(":")[1], date_format)
    
    # 시간 범위에 맞는 인덱스 찾기
    matching_indices = [
        i
        for i, date in enumerate(data["times"])
        if (not isinstance(date, str)) and (target_date_start < date < target_date_end)
    ]
    
    filtered_count = len(matching_indices)
    logger.info(f"시간 필터링 후 문서 수: {filtered_count}, 기간: {time_bound}")
    
    # 필터링 결과가 너무 적은 경우 경고
    if filtered_count < 50 and filtered_count < original_count * 0.1:
        logger.warning(f"시간 필터로 인해 문서가 크게 줄었습니다: {original_count} → {filtered_count}")
    
    # 필터링된 데이터 구성
    filtered_data = {}
    filtered_data["file_names"] = [data["file_names"][i] for i in matching_indices]
    filtered_data["titles"] = [data["titles"][i] for i in matching_indices]
    filtered_data["times"] = [data["times"][i] for i in matching_indices]
    filtered_data["chunk_ids"] = [data["chunk_ids"][i] for i in matching_indices]
    filtered_data["file_path"] = [data["file_path"][i] for i in matching_indices]
    
    # 벡터 데이터 처리
    if isinstance(data["vectors"], torch.Tensor):
        filtered_data["vectors"] = data["vectors"][matching_indices]
    else:
        filtered_data["vectors"] = [data["vectors"][i] for i in matching_indices]
    
    # 텍스트 데이터 처리
    filtered_data["texts"] = [data["texts"][i] for i in matching_indices]
    filtered_data["texts_short"] = [data["texts_short"][i] for i in matching_indices]
    filtered_data["texts_vis"] = [data["texts_vis"][i] for i in matching_indices]
    
    return filtered_data

@time_tracker
def expand_time_range_if_needed(
    time_bound: str, 
    data: Dict[str, Any], 
    min_docs: int = 50
) -> Dict[str, Any]:
    """
    검색 결과가 너무 적은 경우 시간 범위를 자동으로 확장
    
    Args:
        time_bound: 시간 범위
        data: 원본 데이터
        min_docs: 최소 필요 문서 수
        
    Returns:
        Dict[str, Any]: 필터링된 데이터 (필요시 확장된 범위 적용)
    """
    # 전체 기간 선택 시 원본 반환
    if time_bound == "all" or time_bound == "1900-01-01:2099-01-01":
        logger.info("전체 기간 사용")
        return data
    
    # 초기 필터링 시도
    filtered_data = sort_by_time(time_bound, data)
    filtered_count = len(filtered_data.get("times", []))
    
    # 충분한 문서가 있으면 그대로 반환
    if filtered_count >= min_docs:
        logger.info(f"원래 범위로 충분한 문서 확보: {filtered_count}개")
        return filtered_data
    
    logger.info(f"문서 수 부족: {filtered_count}개 (최소 필요: {min_docs}개) - 범위 확장 시도")
    
    # 시간 범위 파싱
    date_format = "%Y-%m-%d"
    try:
        start_date = datetime.strptime(time_bound.split(":")[0], date_format)
        end_date = datetime.strptime(time_bound.split(":")[1], date_format)
    except Exception as e:
        logger.error(f"날짜 형식 오류: {time_bound}, 오류: {e}")
        return data
    
    # 단계적으로 범위 확장 시도
    expansions = [
        (3, "3개월"),
        (6, "6개월"),
        (12, "1년"),
        (24, "2년"),
        (60, "5년"),
    ]
    
    for months, label in expansions:
        # 확장된 범위 계산
        new_start = start_date - timedelta(days=30 * months // 2)
        new_end = end_date + timedelta(days=30 * months // 2)
        
        new_range = f"{new_start.strftime(date_format)}:{new_end.strftime(date_format)}"
        logger.info(f"시간 범위 {label} 확장 시도: {new_range}")
        
        # 확장된 범위로 필터링
        expanded_data = sort_by_time(new_range, data)
        expanded_count = len(expanded_data.get("times", []))
        
        # 충분한 문서를 찾았으면 반환
        if expanded_count >= min_docs:
            logger.info(f"{label} 확장으로 {expanded_count}개 문서 확보")
            return expanded_data
    
    # 모든 확장 시도 실패 시 전체 데이터 반환
    logger.warning("모든 확장 시도 실패, 전체 데이터셋 사용")
    return data 