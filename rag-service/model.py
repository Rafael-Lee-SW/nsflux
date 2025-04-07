import os
import torch
import asyncio
from transformers import AutoModel, AutoTokenizer
from loguru import logger
from utils import time_tracker
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading

# 전역 스레드 풀 및 락 설정
thread_pool = ThreadPoolExecutor(max_workers=4)
model_lock = threading.Lock()

class EmbeddingModel:
    """임베딩 모델 관리 클래스"""
    
    def __init__(self, model_id, cache_dir, batch_size=8, max_workers=4):
        """
        임베딩 모델 초기화
        
        Args:
            model_id: 모델 ID
            cache_dir: 캐시 디렉토리
            batch_size: 배치 처리 크기
            max_workers: 최대 작업자 수
        """
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"임베딩 모델 초기화: device={self.device}, batch_size={batch_size}, max_workers={max_workers}")
        
    @time_tracker
    def load(self):
        """모델 및 토크나이저 로드"""
        logger.info(f"임베딩 모델 로드 시작: {self.model_id}")
        
        # Hugging Face 토큰 가져오기
        token = os.getenv("HF_TOKEN_PATH")
        if token is not None and not token.startswith("hf_"):
            if os.path.exists(token) and os.path.isfile(token):
                try:
                    with open(token, "r") as f:
                        token = f.read().strip()
                except Exception as e:
                    logger.warning(f"토큰 파일 읽기 실패: {e}")
                    token = None
            else:
                logger.warning(f"토큰 파일이 존재하지 않음: {token}")
                token = None
        
        if token is None or token == "":
            logger.warning("HF_TOKEN이 설정되지 않음. 게이트 모델 접근이 실패할 수 있습니다.")
            token = None
        
        try:
            # 모델 로드 - 최적화 옵션 추가
            self.model = AutoModel.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                token=token,
                device_map="auto",
                torch_dtype=torch.float16,  # FP16 사용으로 메모리 절약 및 속도 향상
                low_cpu_mem_usage=True,     # CPU 메모리 사용량 최적화
            )
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                token=token,
            )
            
            # 모델 설정
            self.model.eval()
            self.tokenizer.model_max_length = 8192
            
            # 모델 최적화
            if self.device == "cuda":
                # CUDA 최적화
                torch.backends.cudnn.benchmark = True
                # 모델을 CUDA로 이동
                self.model = self.model.to(self.device)
            
            logger.info("임베딩 모델 로드 완료")
            return True
            
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            return False
    
    @time_tracker
    def embed(self, text):
        """
        텍스트를 임베딩 벡터로 변환 (동기 방식)
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            torch.Tensor: 임베딩 벡터
        """
        if self.model is None or self.tokenizer is None:
            logger.error("모델이 로드되지 않음")
            return None
        
        logger.info(f"임베딩 시작: '{text}'")
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text, 
                    max_length=4096, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                # 마지막 시퀀스 토큰을 임베딩으로 사용
                embeddings = outputs.last_hidden_state[:,-1].cpu()
            
            logger.info("임베딩 완료")
            return embeddings
            
        except Exception as e:
            logger.error(f"임베딩 실패: {e}")
            return None
    
    @time_tracker
    async def embed_async(self, text):
        """
        텍스트를 임베딩 벡터로 변환 (비동기 방식)
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            torch.Tensor: 임베딩 벡터
        """
        if self.model is None or self.tokenizer is None:
            logger.error("모델이 로드되지 않음")
            return None
        
        logger.info(f"비동기 임베딩 시작: '{text}'")
        
        # 스레드 풀에서 실행하여 메인 스레드를 차단하지 않도록 함
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.thread_pool, 
            partial(self._embed_internal, text)
        )
        
        return result
    
    def _embed_internal(self, text):
        """
        내부 임베딩 함수 (스레드 풀에서 실행)
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            torch.Tensor: 임베딩 벡터
        """
        try:
            with model_lock:  # 모델 접근 시 락 사용
                with torch.no_grad():
                    inputs = self.tokenizer(
                        text, 
                        max_length=4096, 
                        padding="max_length", 
                        truncation=True, 
                        return_tensors="pt"
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    # 마지막 시퀀스 토큰을 임베딩으로 사용
                    embeddings = outputs.last_hidden_state[:,-1].cpu()
                
                return embeddings
                
        except Exception as e:
            logger.error(f"내부 임베딩 실패: {e}")
            return None
    
    @time_tracker
    async def embed_batch_async(self, texts):
        """
        여러 텍스트를 배치로 임베딩 (비동기 방식)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            List[torch.Tensor]: 임베딩 벡터 리스트
        """
        if self.model is None or self.tokenizer is None:
            logger.error("모델이 로드되지 않음")
            return [None] * len(texts)
        
        logger.info(f"배치 임베딩 시작: {len(texts)}개 텍스트")
        
        # 배치 크기로 분할
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        results = []
        
        for batch in batches:
            # 각 배치를 비동기적으로 처리
            batch_results = await asyncio.gather(
                *[self.embed_async(text) for text in batch]
            )
            results.extend(batch_results)
        
        logger.info(f"배치 임베딩 완료: {len(results)}개 결과")
        return results
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False) 