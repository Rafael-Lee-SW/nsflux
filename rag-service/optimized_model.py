import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import torch
from loguru import logger
import asyncio
from functools import partial
import ray
from typing import List, Dict, Any, Tuple, Optional, Set

class ONNXEmbeddingModel:
    """Optimized embedding model using ONNX Runtime with continuous batching"""
    
    def __init__(self, model_id, batch_size=8, max_workers=4):
        """
        Initialize the ONNX embedding model
        
        Args:
            model_id: Path to the ONNX model directory or model ID
            batch_size: Batch size for processing
            max_workers: Maximum number of Ray workers
        """
        self.model_id = model_id
        self.model_path = model_id  # For backward compatibility
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.model = None
        self.tokenizer = None
        self.session_options = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_dim = 768  # Default embedding dimension
        
        # 연속 배치 처리 관련 속성
        self.request_queue = asyncio.Queue()
        self.active_tasks = set()
        self.max_concurrent_tasks = batch_size
        self.batch_processor_task = None
        
        # Configure Ray if not already initialized
        if not ray.is_initialized():
            try:
                ray.init(num_cpus=max_workers, ignore_reinit_error=True)
            except Exception as e:
                logger.error(f"Ray 초기화 실패: {e}")
            
        logger.info(f"ONNX Embedding model initialized: device={self.device}, batch_size={batch_size}, max_workers={max_workers}")
    
    def load(self):
        """Load the ONNX model and tokenizer"""
        logger.info(f"Loading ONNX model from {self.model_path}")
        
        try:
            # Configure ONNX Runtime session
            self.session_options = ort.SessionOptions()
            self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session_options.intra_op_num_threads = self.max_workers
            
            # Set execution provider
            if self.device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            # Load model
            model_file = os.path.join(self.model_path, "model.onnx")
            self.model = ort.InferenceSession(
                model_file, 
                sess_options=self.session_options,
                providers=providers
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if hasattr(self.tokenizer, 'model_max_length'):
                self.tokenizer.model_max_length = 8192
                
            # 임베딩 차원 추출
            outputs_meta = self.model.get_outputs()
            if outputs_meta and len(outputs_meta) > 0:
                # 출력 shape에서 차원 추출
                output_shape = outputs_meta[0].shape
                if output_shape and len(output_shape) > 1:
                    self.embedding_dim = output_shape[-1]  # 마지막 차원이 임베딩 차원
                    logger.info(f"모델의 임베딩 차원 탐지: {self.embedding_dim}")
            
            # 배치 프로세서 시작
            self.start_batch_processor()
                
            logger.info("ONNX model and tokenizer loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {str(e)}")
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
            # 실제 임베딩 처리는 ThreadPoolExecutor를 통해 비동기적으로 수행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.embed, text)
            future.set_result(result)
            logger.info(f"쿼리 처리 완료: '{text[:30]}...'")
        except Exception as e:
            logger.error(f"쿼리 처리 중 오류: {e}")
            future.set_exception(e)