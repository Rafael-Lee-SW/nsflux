import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import torch
from loguru import logger
import asyncio
from functools import partial
import ray

class ONNXEmbeddingModel:
    """Optimized embedding model using ONNX Runtime"""
    
    def __init__(self, model_path, batch_size=8, max_workers=4):
        """
        Initialize the ONNX embedding model
        
        Args:
            model_path: Path to the ONNX model directory
            batch_size: Batch size for processing
            max_workers: Maximum number of Ray workers
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.model = None
        self.tokenizer = None
        self.session_options = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configure Ray if not already initialized
        if not ray.is_initialized():
            ray.init(num_cpus=max_workers)
            
        logger.info(f"ONNX Embedding model initialized: device={self.device}, batch_size={batch_size}")
    
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
                
            logger.info("ONNX model and tokenizer loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {str(e)}")
            return False
    
    @ray.remote
    def _embed_ray(model_path, text, session_options, device):
        """Ray remote function for parallel embedding"""
        # Load model within the Ray worker
        if device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            
        model = ort.InferenceSession(
            os.path.join(model_path, "model.onnx"),
            sess_options=session_options,
            providers=providers
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Tokenize
        inputs = tokenizer(
            text, 
            max_length=4096, 
            padding="max_length", 
            truncation=True, 
            return_tensors="np"
        )
        
        # Run inference
        outputs = model.run(
            None, 
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
        )
        
        # Get embedding from the last hidden state
        # Assuming outputs[0] is the last hidden state with shape [batch_size, seq_len, hidden_size]
        embeddings = outputs[0][:, -1, :]  # Get the embedding of the last token
        
        return embeddings
    
    def embed(self, text):
        """Synchronous embedding"""
        logger.info(f"Embedding text: '{text}'")
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text, 
                max_length=4096, 
                padding="max_length", 
                truncation=True, 
                return_tensors="np"
            )
            
            # Run inference
            outputs = self.model.run(
                None, 
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                }
            )
            
            # Get embedding from the last hidden state
            embeddings = outputs[0][:, -1, :]  # Get the embedding of the last token
            
            # Convert to tensor for compatibility
            embeddings_tensor = torch.tensor(embeddings)
            
            logger.info("Embedding completed")
            return embeddings_tensor
            
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            return None
    
    async def embed_async(self, text):
        """Asynchronous embedding"""
        logger.info(f"Async embedding text: '{text}'")
        
        # Run in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.embed, text)
        
        return result
    
    async def embed_batch_async(self, texts):
        """Process a batch of texts in parallel using Ray"""
        logger.info(f"Batch embedding {len(texts)} texts")
        
        # Split into batches
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        results = []
        
        for batch in batches:
            # Process batch in parallel with Ray
            futures = [
                self._embed_ray.remote(
                    self.model_path, text, self.session_options, self.device
                ) for text in batch
            ]
            
            # Get results
            batch_results = ray.get(futures)
            
            # Convert to tensors
            batch_tensors = [torch.tensor(emb) for emb in batch_results]
            results.extend(batch_tensors)
        
        logger.info(f"Batch embedding completed for {len(results)} texts")
        return results 