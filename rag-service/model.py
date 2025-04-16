import os
import torch
import asyncio
from transformers import AutoTokenizer
from loguru import logger
from utils import time_tracker
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import deque

# Conditional import for vLLM
try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    VLLM_AVAILABLE = True
except ImportError:
    logger.warning("vLLM not available, using standard transformers API instead")
    VLLM_AVAILABLE = True

class EmbeddingModel:
    """vLLM 기반 임베딩 모델 관리 클래스"""
    
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_vllm = VLLM_AVAILABLE and self.device == "cuda"  # vLLM requires GPU
        self.embedding_dim = 4096  # Default embedding dimension
        # vLLM 문제 발생 시 표준 모드로 강제 전환하는 플래그
        self.force_standard_mode = False
        
        # 연속 배치 처리 관련 속성 추가
        self.request_queue = asyncio.Queue()
        self.active_tasks = set()
        self.max_concurrent_tasks = batch_size
        self.batch_processor_task = None
        
        logger.info(f"임베딩 모델 초기화: device={self.device}, batch_size={batch_size}, max_workers={max_workers}, use_vllm={self.use_vllm}")
        
    @time_tracker
    def load(self):
        """모델 및 토크나이저 로드"""
        logger.info(f"임베딩 모델 로드 시작: {self.model_id}")
        
        token = None
        token_path = os.getenv("HF_TOKEN_PATH")
        if token_path and os.path.exists(token_path) and os.path.isfile(token_path):
            try:
                with open(token_path, "r") as f:
                    token = f.read().strip()
                logger.info(f"토큰 파일 로드 성공: {token_path}")
            except Exception as e:
                logger.warning(f"토큰 파일 읽기 실패 ({token_path}): {e}")
                token = None
        else:
            logger.warning(f"HF_TOKEN_PATH 환경 변수에 지정된 토큰 파일이 없거나 유효하지 않음: {token_path}. HF_HOME ({os.getenv('HF_HOME', '기본값')}) 캐시 확인.")

        if token is None or token == "":
            logger.warning("사용할 HF 토큰을 찾지 못했습니다. 공개 모델만 접근 가능합니다.")
            token = None # Ensure token is None if not found

        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                token=token,
            )
            
            # 토크나이저 설정
            if hasattr(self.tokenizer, 'model_max_length'):
                self.tokenizer.model_max_length = 8192
            
            # vLLM 또는 표준 모델 로드
            if self.use_vllm:
                # vLLM 엔진 인자 설정
                try:
                    logger.info("vLLM AsyncLLMEngine 로드 시도")
                    
                    # AsyncEngineArgs 생성
                    engine_args = AsyncEngineArgs(
                        model=self.model_id,
                        dtype="bfloat16",
                        trust_remote_code=True,
                        tensor_parallel_size=min(torch.cuda.device_count(), 1),
                        gpu_memory_utilization=0.95,
                        max_num_batched_tokens=32768,
                    )
                    
                    # AsyncLLMEngine 생성
                    self.model = AsyncLLMEngine.from_engine_args(engine_args)
                    logger.info("vLLM AsyncLLMEngine 로드 성공")
                except Exception as e:
                    logger.error(f"vLLM AsyncLLMEngine 로드 실패: {e}")
                    logger.info("표준 Transformers 모델로 대체 시도")
                    self.use_vllm = False
            
            # 표준 모델 로드 (vLLM 실패 또는 사용 불가시)
            if not self.use_vllm:
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    self.model_id,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    token=token,
                )
                
                # 임베딩 차원 확인
                if hasattr(self.model.config, 'hidden_size'):
                    self.embedding_dim = self.model.config.hidden_size
                
                # GPU 사용 설정
                if torch.cuda.is_available():
                    self.model = self.model.to('cuda')
                    logger.info("GPU 사용 설정 완료")
                else:
                    logger.info("CPU 사용 설정 완료")
            
            # 배치 프로세서 시작
            self.start_batch_processor()
            
            logger.info("임베딩 모델 로드 완료")
            return True
            
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            return False
    
    def start_batch_processor(self):
        """연속 배치 처리기 시작"""
        if self.batch_processor_task is None or self.batch_processor_task.done():
            self.batch_processor_task = asyncio.create_task(self.continuous_batch_processor())
            logger.info("연속 배치 처리기 시작됨")
    
    # -------------------------------------------------------------------------
    # CONTINUOUOS BATCH PROCESSOR - 연속 배치 처리
    # -------------------------------------------------------------------------
    async def continuous_batch_processor(self):
        """
        연속적인 배치 처리 - 최대 max_concurrent_tasks개의 요청이 항상 동시에 처리되도록 함
        """
        logger.info(f"연속 배치 처리기 시작: 최대 동시 작업 수={self.max_concurrent_tasks}")
        try:
            while True:
                # 사용 가능한 슬롯 확인
                available_slots = self.max_concurrent_tasks - len(self.active_tasks)
                
                if available_slots > 0:
                    try:
                        # 새 요청 받기 (타임아웃 0.01초)
                        request_obj, fut = await asyncio.wait_for(self.request_queue.get(), timeout=0.01)

                        # 비동기 처리 작업 생성
                        task = asyncio.create_task(
                            self._process_single_query(request_obj, fut)
                        )
                        
                        # 작업 완료 시 활성 목록에서 제거하는 콜백 설정
                        task.add_done_callback(
                            lambda t, task_ref=task: self.active_tasks.discard(task_ref)
                        )
                        
                        # 활성 작업 목록에 추가
                        self.active_tasks.add(task)
                        
                        logger.info(f"[연속 배치] +1 요청 => 현재 활성 작업 {len(self.active_tasks)}/{self.max_concurrent_tasks}")
                    
                    except asyncio.TimeoutError:
                        # 새 요청이 없으면 잠시 대기
                        await asyncio.sleep(0.01)
                else:
                    # 모든 슬롯이 사용 중이면 작업 완료될 때까지 대기
                    if self.active_tasks:
                        # 작업 중 하나라도 완료될 때까지 대기
                        done, pending = await asyncio.wait(
                            self.active_tasks, 
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        # 완료된 작업 확인
                        for task in done:
                            try:
                                await task
                            except Exception as e:
                                logger.error(f"작업 실패: {e}")
                        
                        logger.info(f"[연속 배치] 완료된 작업: {len(done)} => 현재 활성 작업 {len(self.active_tasks)}/{self.max_concurrent_tasks}")
                    else:
                        await asyncio.sleep(0.01)
                        
        except asyncio.CancelledError:
            logger.info("배치 처리기가 취소되었습니다.")
        except Exception as e:
            logger.error(f"배치 처리기 오류: {e}")
    
    async def _process_single_query(self, text, future):
        """
        단일 쿼리 처리 (배치 처리기에서 호출됨)
        
        Args:
            text: 임베딩할 텍스트
            future: 결과를 설정할 Future 객체
        """
        try:
            logger.info(f"쿼리 처리 시작: '{text[:30]}...'")
            result = await self._actual_embed_async(text)
            future.set_result(result)
            logger.info(f"쿼리 처리 완료: '{text[:30]}...'")
        except Exception as e:
            logger.error(f"쿼리 처리 중 오류: {e}")
            future.set_exception(e)
    
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
            return torch.zeros(self.embedding_dim, dtype=torch.float32)
        
        logger.info(f"비동기 임베딩 요청: '{text[:30]}...'")
        
        # Future 객체 생성 및 큐에 추가
        future = asyncio.Future()
        await self.request_queue.put((text, future))
        
        # 결과 대기
        return await future
    
    async def _actual_embed_async(self, text):
        """
        실제 임베딩 수행 (내부 호출용)
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            torch.Tensor: 임베딩 벡터
        """
        try:
            # 표준 모드로 강제 전환된 경우 vLLM을 사용하지 않음
            if self.force_standard_mode:
                logger.info("표준 모드로 강제 전환되어 있어 vLLM을 사용하지 않습니다.")
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._embed_sync, text)
                
            # vLLM 사용시 처리 방법
            if self.use_vllm:
                try:
                    # vLLM을 사용한 임베딩 생성
                    sampling_params = SamplingParams(
                        temperature=0.0,  # 결정적 출력을 위해 temperature 0으로 설정
                        max_tokens=1,     # 임베딩을 위해 최소한의 토큰만 생성
                    )
                    
                    # 고유 요청 ID 생성
                    request_id = f"req_{hash(text)}"
                    
                    # AsyncLLMEngine의 generate 메서드는 직접적으로 호출하는 대신 비동기 for 루프로 처리
                    outputs = []
                    async_generator = self.model.generate(prompt=text, sampling_params=sampling_params, request_id=request_id)
                    
                    # Note: This is an async generator, not a callable
                    async for output in async_generator:
                        outputs.append(output)
                    
                    # 출력이 없는 경우 처리
                    if not outputs:
                        logger.warning("vLLM 출력이 없습니다. 표준 방식으로 전환합니다.")
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(None, self._embed_sync, text)
                    
                    # 마지막 출력 사용
                    final_output = outputs[-1]
                    
                    # 디버그 정보 출력
                    logger.info(f"vLLM final output keys: {dir(final_output)}")
                    
                    # PoolingOutput 객체에서 임베딩 추출
                    if hasattr(final_output, 'outputs'):
                        outputs_info = final_output.outputs
                        logger.info(f"outputs 타입: {type(outputs_info)}")
                        
                        # PoolingOutput 객체 처리
                        if hasattr(outputs_info, 'embedding'):
                            logger.info("embedding 속성을 발견했습니다 - 이를 임베딩으로 사용합니다.")
                            embedding_data = outputs_info.embedding
                            
                            # 임베딩 데이터 형식 검사 및 변환
                            if embedding_data is not None:
                                logger.info(f"임베딩 데이터 타입: {type(embedding_data)}, 형태: {np.array(embedding_data).shape if hasattr(embedding_data, '__len__') else 'unknown'}")
                                
                                # 차원 축소 또는 확장이 필요한지 확인
                                embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32)
                                if embedding_tensor.shape[-1] != self.embedding_dim:
                                    logger.warning(f"임베딩 차원 불일치: {embedding_tensor.shape[-1]} vs {self.embedding_dim}. 임베딩을 {self.embedding_dim}차원으로 조정합니다.")
                                    
                                    # 차원이 클 경우 PCA와 유사하게 처리 (가장 중요한 차원만 유지)
                                    if embedding_tensor.shape[-1] > self.embedding_dim:
                                        return embedding_tensor[:self.embedding_dim]
                                    
                                    # 차원이 작을 경우 패딩
                                    else:
                                        padded = torch.zeros(self.embedding_dim, dtype=torch.float32)
                                        padded[:embedding_tensor.shape[-1]] = embedding_tensor
                                        return padded
                                
                                return embedding_tensor
                        
                        logger.warning("outputs에 embedding 속성이 없습니다.")
                    
                    # outputs 속성이 없는 경우
                    logger.warning("vLLM 출력에서 임베딩을 추출할 수 없습니다. 표준 방식으로 전환합니다.")
                    
                    # 표준 방식으로 전환
                    self.force_standard_mode = True  # 이후 호출에서도 표준 모드 사용
                    logger.info("표준 Transformer 방식으로 전환합니다. (이후 모든 요청에 적용)")
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, self._embed_sync, text)
                        
                except Exception as e:
                    logger.error(f"vLLM 임베딩 처리 중 오류: {e}")
                    logger.info("표준 transformer 방식으로 임베딩 시도")
                    # 오류 발생 시 향후 모든 요청을 표준 모드로 처리
                    self.force_standard_mode = True
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, self._embed_sync, text)
            
            # 표준 transformers 사용시 처리 방법
            else:
                # 비동기 처리로 전환
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._embed_sync, text)
                
        except Exception as e:
            logger.error(f"임베딩 실패: {e}")
            # 실패 시 기본 임베딩 반환
            return torch.zeros(self.embedding_dim, dtype=torch.float32)
    
    def _embed_sync(self, text):
        """동기식 임베딩 처리 (run_in_executor에서 사용)"""
        try:
            if self.model is None or self.tokenizer is None:
                logger.error("모델이 로드되지 않음")
                return torch.zeros((1, self.embedding_dim), dtype=torch.float32)
                
            # vLLM 모델은 호출할 수 없으므로 표준 모델 호출 방식만 지원
            if self.use_vllm and not hasattr(self.model, '__call__'):
                logger.error("vLLM 모델은 동기식 호출을 지원하지 않습니다. transformer 모델이 필요합니다.")
                return torch.zeros((1, self.embedding_dim), dtype=torch.float32)
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    text, 
                    max_length=4096, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                )
                
                # GPU로 이동
                if torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                    
                outputs = self.model(**inputs)
                # Last Sequence Token is used as Embedding
                embeddings = outputs.last_hidden_state[:, -1].cpu()
                
                # 임베딩 차원이 데이터와 일치하는지 확인
                if embeddings.shape[-1] != self.embedding_dim:
                    logger.warning(f"임베딩 차원 불일치: {embeddings.shape[-1]} vs {self.embedding_dim}. 임베딩을 {self.embedding_dim}차원으로 조정합니다.")
                    
                    # 차원 조정 (확장 또는 축소)
                    if embeddings.shape[-1] > self.embedding_dim:
                        embeddings = embeddings[:, :self.embedding_dim]
                    else:
                        padded = torch.zeros((1, self.embedding_dim), dtype=torch.float32)
                        padded[:, :embeddings.shape[-1]] = embeddings
                        embeddings = padded
                
                return embeddings[0]  # 첫 번째 배치 항목만 반환
        except Exception as e:
            logger.error(f"동기식 임베딩 처리 실패: {e}")
            return torch.zeros(self.embedding_dim, dtype=torch.float32)
    
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
            return [torch.zeros(self.embedding_dim, dtype=torch.float32)] * len(texts)
        
        logger.info(f"배치 임베딩 시작: {len(texts)}개 텍스트")
        
        # 비동기 작업 생성
        futures = []
        for text in texts:
            future = asyncio.Future()
            await self.request_queue.put((text, future))
            futures.append(future)
        
        # 모든 작업 완료 대기
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # 오류 처리 및 결과 가공
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"텍스트 {i} 처리 실패: {result}")
                processed_results.append(torch.zeros(self.embedding_dim, dtype=torch.float32))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_with_index(self, index, text, results_list):
        """특정 인덱스에 대한 텍스트 처리 (결과는 결과 리스트에 직접 저장)"""
        try:
            embedding = await self._actual_embed_async(text)
            results_list[index] = embedding
        except Exception as e:
            logger.error(f"텍스트 {index} 처리 실패: {e}")
            results_list[index] = torch.zeros(self.embedding_dim, dtype=torch.float32)
    
    def _embed_batch_sync(self, texts):
        """배치 텍스트 동기식 임베딩 (asyncio.to_thread에서 사용)"""
        embeddings = []
        
        # 모델이 로드되지 않았거나 vLLM인 경우 기본값 반환
        if self.model is None or self.tokenizer is None or (self.use_vllm and not hasattr(self.model, '__call__')):
            logger.error("동기식 배치 임베딩을 위한 유효한 모델이 없습니다")
            return [torch.zeros(self.embedding_dim, dtype=torch.float32)] * len(texts)
            
        with torch.no_grad():
            for text in texts:
                try:
                    inputs = self.tokenizer(
                        text, 
                        max_length=4096, 
                        padding="max_length", 
                        truncation=True, 
                        return_tensors="pt"
                    )
                    
                    # GPU로 이동
                    if torch.cuda.is_available():
                        inputs = {k: v.to('cuda') for k, v in inputs.items()}
                        
                    outputs = self.model(**inputs)
                    # Last Sequence Token is used as Embedding
                    embedding = outputs.last_hidden_state[:, -1].cpu()
                    
                    # 차원 조정
                    if embedding.shape[-1] != self.embedding_dim:
                        if embedding.shape[-1] > self.embedding_dim:
                            embedding = embedding[0, :self.embedding_dim]
                        else:
                            padded = torch.zeros(self.embedding_dim, dtype=torch.float32)
                            padded[:embedding.shape[-1]] = embedding[0]
                            embedding = padded
                    else:
                        embedding = embedding[0]  # 첫 번째 배치 항목만 추출 (배치 크기가 1)
                        
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"텍스트 임베딩 실패: {e}")
                    embeddings.append(torch.zeros(self.embedding_dim, dtype=torch.float32))
                
        return embeddings
    
    def stop_batch_processor(self):
        """배치 처리기 중지"""
        if self.batch_processor_task and not self.batch_processor_task.done():
            self.batch_processor_task.cancel()
            logger.info("배치 처리기 중지 요청됨")
    
    def __del__(self):
        """리소스 정리"""
        self.stop_batch_processor()
        if hasattr(self, 'model'):
            del self.model