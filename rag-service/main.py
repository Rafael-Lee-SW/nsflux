from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import json
import asyncio
from datetime import datetime, timedelta, timezone
import uuid                                # ★ NEW
import numpy as np
from dotenv import load_dotenv
from loguru import logger
import httpx
import torch
import ray
from contextlib import asynccontextmanager
from config import settings
from utils import load_data, random_seed
from retrieval import retrieve, sort_by_time, expand_time_range_if_needed

# Conditional imports based on settings
if settings.USE_ONNX:
    from optimized_model import ONNXEmbeddingModel as EmbeddingModel
else:
    from model import EmbeddingModel

# ──────────────────────────────────────────────────────────────────────────────
# req_id 중복 방지용 전역 상태
# ──────────────────────────────────────────────────────────────────────────────
REQ_ID_TTL_SECONDS = 90                          # 중복 체크 유지 시간(초)
_recent_req_ids: dict[str, float] = {}           # {req_id: last_seen_timestamp}
_req_id_lock = asyncio.Lock()                    # 동시 요청 보호용 Lock


async def _register_req_id(req_id: str) -> None:
    """
    중복 req_id가 들어오면 HTTPException(409) 발생.
    TTL이 지난 req_id는 자동 제거.
    """
    now = datetime.now(timezone.utc).timestamp()
    async with _req_id_lock:
        # 만료된 항목 정리
        expired = [
            rid for rid, ts in _recent_req_ids.items()
            if now - ts > REQ_ID_TTL_SECONDS
        ]
        for rid in expired:
            _recent_req_ids.pop(rid, None)

        # 중복 확인
        if req_id in _recent_req_ids:
            raise HTTPException(
                status_code=409,
                detail=f"Duplicate req_id detected: {req_id}"
            )
        # 신규 등록
        _recent_req_ids[req_id] = now
# ──────────────────────────────────────────────────────────────────────────────

# 환경 변수 로드
load_dotenv()

# 전역 변수로 모델과 데이터 저장
model = None
data = None
embedding_cache = {}

# 모델 초기화 함수
def initialize_model():
    global model
    try:
        logger.info("모델 초기화 시작...")
        
        # 임베딩 모델 생성
        model = EmbeddingModel(
            model_id=settings.MODEL_NAME,
            cache_dir=settings.CACHE_DIR,
            batch_size=settings.BATCH_SIZE,
            max_workers=settings.MAX_WORKERS
        )
        
        # 모델 로드
        if model.load():
            logger.info("모델 초기화 완료")
            return True
        else:
            logger.error("모델 로드 실패")
            return False
            
    except Exception as e:
        logger.error(f"모델 초기화 중 오류 발생: {str(e)}")
        return False
    
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서비스 시작 및 종료 시 실행되는 lifespan 이벤트 핸들러"""
    global model, data
    
    # 서비스 시작 시 초기화 작업 수행
    try:
        # 랜덤 시드 설정
        random_seed(42)  # Fixed seed for reproducibility
        
        # Ray 초기화 (설정에 따라 조건부 실행)
        if settings.USE_RAY:
            if not ray.is_initialized():
                try:
                    ray.init(
                        address=settings.RAY_ADDRESS,
                        num_cpus=settings.RAY_NUM_CPUS,
                        num_gpus=settings.RAY_NUM_GPUS if settings.RAY_NUM_GPUS > 0 else None,
                        ignore_reinit_error=True
                    )
                    logger.info(f"Ray initialized with {settings.RAY_NUM_CPUS} CPUs and {settings.RAY_NUM_GPUS} GPUs")
                except Exception as e:
                    logger.error(f"Ray initialization failed: {e}")
                    logger.info("Proceeding without Ray")
        else:
            logger.info("Ray 사용 비활성화됨")
        
        # 모델 초기화
        if not initialize_model():
            logger.error("Failed to initialize model, service may not function correctly")
        
        # 데이터 로드
        data = load_data(settings.DATA_PATH, settings.IMAGE_BASE_PATH)
        if data is None:
            logger.error(f"Failed to load data from {settings.DATA_PATH}")
        
        logger.info("서비스 초기화 완료")
        
    except Exception as e:
        logger.error(f"서비스 초기화 실패: {str(e)}")
        logger.warning("서비스가 제한된 기능으로 실행됩니다")
    
    yield
    
    # 서비스 종료 시 정리 작업
    logger.info("서비스 종료 중...")
    
    # 모델 배치 프로세서 중지
    if model:
        if hasattr(model, 'stop_batch_processor'):
            model.stop_batch_processor()
            logger.info("모델 배치 프로세서 중지됨")
    
    # Ray 종료
    if settings.USE_RAY and ray.is_initialized():
        try:
            ray.shutdown()
            logger.info("Ray shut down")
        except Exception as e:
            logger.error(f"Ray shutdown error: {e}")
    
    logger.info("서비스 종료 완료")

# FastAPI 앱 초기화
app = FastAPI(
    title="RAG Service",
    description="Retrieval-Augmented Generation Service with Multi-Processing Support",
    version="1.0.0",
    lifespan=lifespan
)

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────────────────────────────────────
class RetrieveRequest(BaseModel):
    query: str
    top_n: int = 10
    time_bound: str = "all"
    min_docs: int = 50
    req_id: Optional[str] = None               # ★ NEW : 클라이언트가 지정할 수도 있음

class Document(BaseModel):
    file_name: str
    title: str
    contents: List[dict]
    chunk_id: int
    file_path: Optional[str] = None
    text_short: Optional[str] = None

class RetrieveResponse(BaseModel):
    documents: str
    documents_list: List[Document]
    req_id: str                                # ★ NEW : 실제 사용된 req_id 반환

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """서비스 상태 확인 엔드포인트"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": data is not None,
        "ray_initialized": ray.is_initialized() if settings.USE_RAY else False
    }

@app.post("/api/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    """문서 검색 엔드포인트 (req_id 중복 방지 포함)"""
    try:
        # 모델과 데이터 확인
        if model is None or data is None:
            raise HTTPException(
                status_code=503,
                detail="서비스가 아직 초기화되지 않았습니다"
            )

        # ───── req_id 중복 체크 ─────
        # 클라이언트가 주지 않으면 새로 생성
        req_id = request.req_id or uuid.uuid4().hex
        await _register_req_id(req_id)

        # 시간 기반 필터링
        filtered_data = sort_by_time(request.time_bound, data)
        
        # 필요한 경우 시간 범위 확장
        if len(filtered_data["times"]) < request.min_docs:
            filtered_data = expand_time_range_if_needed(
                request.time_bound,
                data,
                request.min_docs
            )
        
        # 문서 검색 수행 (비동기 방식)
        documents, documents_list = await retrieve(
            query=request.query,
            data=filtered_data,
            top_n=request.top_n,
            embed_model=model,
            embed_tokenizer=model.tokenizer
        )
        
        # Document 객체로 변환
        formatted_documents = []
        for doc in documents_list:
            try:
                formatted_documents.append(
                    Document(
                        file_name=doc["file_name"],
                        title=doc["title"],
                        contents=doc["contents"],
                        chunk_id=doc["chunk_id"],
                        file_path=doc.get("file_path"),
                        text_short=doc.get("text_short"),
                    )
                )
            except Exception as e:
                logger.warning(f"문서 변환 중 오류 발생 (무시됨): {str(e)}")
                continue
        
        return RetrieveResponse(
            documents=documents,
            documents_list=formatted_documents,
            req_id=req_id                         # 사용된 req_id 반환
        )
        
    except HTTPException:
        # HTTPException 그대로 전달
        raise
    except Exception as e:
        logger.error(f"문서 검색 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"문서 검색 중 오류가 발생했습니다: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
