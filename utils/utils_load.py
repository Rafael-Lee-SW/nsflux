# utils/utils_load.py
import json
import os
import logging
import time
import torch
import numpy as np
import gzip
import builtins
import glob
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import re
import random
import shutil
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
)

# Import vLLM utilities
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# Globals
_cached_data = None  # 벡터 DB 데이터 캐싱
_cached_data_mtime = 0  # 캐시 유효성 확인용 수정 시간

# PyTorch 모델 파일 최소 크기 (fallback 용)
MIN_WEIGHT_SIZE = 10 * 1024 * 1024  # 10MB

# For tracking execution time of functions
from utils.tracking import time_tracker
# For logging system
from utils.log_system import MetricsManager
# Logging
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------
# Function: find_weight_directory - 허깅페이스 권한 문제 해결 후에 잘 사용되지 아니함
# -------------------------------------------------
# Recursively searches for weight files (safetensors or pytorch_model.bin) in a given base path.
# This method Find the files searching the whole directory
# Because, vLLM not automatically find out the model files.
# -------------------------------------------------
@time_tracker
def find_weight_directory(base_path):
    # ---- Recursively searches for weight files in a given base path ----
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if ".safetensors" in file or "pytorch_model.bin" in file:
                file_path = os.path.join(root, file)
                try:
                    if os.path.getsize(file_path) >= MIN_WEIGHT_SIZE:
                        return root, "safetensors" if ".safetensors" in file else "pt"
                    else:
                        logging.debug(
                            f"파일 {file_path}의 크기가 너무 작음: {os.path.getsize(file_path)} bytes"
                        )
                except Exception as ex:
                    logging.debug(f"파일 크기 확인 실패: {file_path} - {ex}")
    return None, None

# -------------------------------------------------
# Function: load_model
# -------------------------------------------------
@time_tracker
def load_model(config):
    # Loads the embedding model and the main LLM model (using vLLM if specified in the config).
    
    # Get the HF token from the environment variable.
    logging.info("Starting model loading...")
    token = os.getenv("HF_TOKEN_PATH")
    # Check if token is likely a file path.
    if token is not None and not token.startswith("hf_"):
        if os.path.exists(token) and os.path.isfile(token):
            try:
                with open(token, "r") as f:
                    token = f.read().strip()
            except Exception as e:
                print("DEBUG: Exception while reading token file:", e)
                logging.warning("Failed to read token from file: %s", e)
                token = None
        else:
            logging.warning("The HF_TOKEN path does not exist: %s", token)
            token = None
    else:
        print("DEBUG: HF_TOKEN appears to be a token string; using it directly:")

    if token is None or token == "":
        logging.warning("HF_TOKEN is not set. Access to gated models may fail.")
        token = None

    # -------------------------------
    # Load the embedding model and tokenizer.
    # -------------------------------
    print("Loading embedding model")
    try:
        embed_model = AutoModel.from_pretrained(
            config.embed_model_id,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
            token=token,
            # device_map="auto"
        )
    except Exception as e:
        raise e
    try:
        embed_tokenizer = AutoTokenizer.from_pretrained(
            config.embed_model_id,
            cache_dir=config.cache_dir,
            trust_remote_code=True,
            token=token,
        )
    except Exception as e:
        raise e
    print(":Embedding tokenizer loaded successfully.")
    embed_model.eval()
    embed_tokenizer.model_max_length = 8192

    # -------------------------------
    # Load the main LLM model via vLLM.
    # -------------------------------
    if config.use_vllm:
        print("vLLM mode enabled. Starting to load main LLM model via vLLM.")
        if config.model.quantization_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("Using 4-bit quantization.")
        elif config.model.quantization_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            print("Using 8-bit quantization.")
        else:
            bnb_config = None
            print("Using pure option of Model(No quantization)")

        local_model_path = os.path.join(
            config.cache_dir, "models--" + config.model_id.replace("/", "--")
        )
        local_model_path = os.path.abspath(local_model_path)

        config_file = os.path.join(local_model_path, "config.json")
        need_patch = False

        if not os.path.exists(config_file):
            os.makedirs(local_model_path, exist_ok=True)
            try:
                hf_config = AutoConfig.from_pretrained(
                    config.model_id,
                    cache_dir=config.cache_dir,
                    trust_remote_code=True,
                    token=token,
                )
            except Exception as e:
                raise e
            # 패치: vocab_size 속성이 없으면 embed_tokenizer의 값을 사용하여 추가
            if not hasattr(hf_config, "vocab_size"):
                print("[MODEL-LOADING] 'vocab_size' 속성이 없어서 기본값으로 추가합니다.")
                hf_config.vocab_size = getattr(embed_tokenizer, "vocab_size", 32000)
            config_dict = hf_config.to_dict()
            # 패치: architectures 속성이 없으면 Gemma2로 기본 설정
            if not config_dict.get("architectures"):
                print("[MODEL-LOADING] Config file의 architectures 정보 없음, Default Gemma2 아키텍처 설정")
                config_dict["architectures"] = ["Gemma2ForCausalLM"]
            # 확인
            print(f"[DEBUG] => Saving HF Config with architectures={config_dict['architectures']}")
                
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f)
        else:
            # 이미 config_file이 존재하는 경우
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            
            # 패치: vocab_size 속성이 없으면 embed_tokenizer의 값을 사용하여 추가
            if "vocab_size" not in config_dict:
                # embed_tokenizer의 vocab_size가 존재하면 사용하고, 없으면 기본값 30522로 설정
                config_dict["vocab_size"] = getattr(embed_tokenizer, "vocab_size", 30522)
                print("[MODEL-LOADING] 'vocab_size' 속성이 없어서 기본값으로 추가합니다:", config_dict["vocab_size"])
            # 패치: architectures 속성이 없으면 Gemma2로 기본 설정
            if not config_dict.get("architectures"):
                print("[MODEL-LOADING] Config file의 architectures 정보 없음, Default Gemma2 아키텍처 설정")
                config_dict["architectures"] = ["Gemma2ForCausalLM"]
            # 확인
            print(f"[DEBUG] => Saving HF Config with architectures={config_dict['architectures']}")

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f)

        weight_dir, weight_format = find_weight_directory(local_model_path)
        if weight_dir is None:
            print("DEBUG: No model weights found. Attempting to download model snapshot.")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"DEBUG: Snapshot download attempt {attempt+1}...")
                    # Attempt to download the model snapshot using the Hugging Face hub function.
                    from huggingface_hub import snapshot_download
                    snapshot_download(config.model_id, cache_dir=config.cache_dir, token=token)
                    break  # If download succeeds, break out of the loop.
                except Exception as e:
                    print(f"DEBUG: Snapshot download attempt {attempt+1} failed:", e)
                    if attempt < max_retries - 1:
                        print("DEBUG: Retrying snapshot download...")
                    else:
                        raise RuntimeError(f"Snapshot download failed after {max_retries} attempts: {e}")
            # After download, try to find the weights again.
            weight_dir, weight_format = find_weight_directory(local_model_path)
            if weight_dir is None:
                raise RuntimeError(f"Unable to find model weights even after snapshot download in {local_model_path}.")

        snapshot_config = os.path.join(weight_dir, "config.json")
        if not os.path.exists(snapshot_config):
            shutil.copy(config_file, snapshot_config)
        engine_args = AsyncEngineArgs(
            model=weight_dir,
            tokenizer=config.model_id,
            download_dir=config.cache_dir,
            trust_remote_code=True,
            config_format="hf",
            load_format=weight_format,
        )
        
        vllm_conf = config.get("vllm", {})
        
        # --- 기존 엔진 설정들
        engine_args.enable_prefix_caching = True
        # engine_args.scheduler_delay_factor = vllm_conf.get("scheduler_delay_factor", 0.1)
        engine_args.enable_chunked_prefill = True
        engine_args.tensor_parallel_size = vllm_conf.get("tensor_parallel_size", 1) # How many use the parllel Multi-GPU
        engine_args.max_num_seqs = vllm_conf.get("max_num_seqs")
        engine_args.max_num_batched_tokens = vllm_conf.get("max_num_batched_tokens", 8192)
        # engine_args.block_size = vllm_conf.get("block_size", 128)
        engine_args.gpu_memory_utilization = vllm_conf.get("gpu_memory_utilization")
        
        # --- 다중 GPU 사용 관련 설정 ---
        if vllm_conf.get("disable_custom_all_reduce", False):
            engine_args.disable_custom_all_reduce = True # For Fixing the Multi GPU problem
        
        # --- 모델의 Context Length ---
        engine_args.max_model_len = vllm_conf.get("max_model_len")
        
        # --- 멀티모달 (Image) 관련 설정 ---
        engine_args.mm_processor_kwargs = vllm_conf.get("mm_processor_kwargs", {"do_pan_and_scan": True})
        engine_args.disable_mm_preprocessor_cache = vllm_conf.get("disable_mm_preprocessor_cache", False)
        engine_args.limit_mm_per_prompt = vllm_conf.get("limit_mm_per_prompt", {"image": 5})
        
        # # --- Metrics (Monitoring) 관련 설정 ---
        # from vllm.config import ObservabilityConfig
        # observability_config = ObservabilityConfig(
        #     show_hidden_metrics=True,
        #     otlp_traces_endpoint=None,
        #     collect_model_forward_time=True,
        #     collect_model_execute_time=True
        # )

        # # Ensure it's properly assigned to engine_args
        # engine_args.observability_config = observability_config
    
        # engine_args.show_hidden_metrics_for_version = "0.7"
        
        engine_args.disable_log_stats = False
    
        print("EngineArgs setting be finished")
        
        try:
            # --- v1 구동 해결책: 현재 스레드가 메인 스레드가 아니면 signal 함수를 임시 패치 ---
            import threading, signal
            if threading.current_thread() is not threading.main_thread():
                original_signal = signal.signal
                signal.signal = lambda s, h: None  # signal 설정 무시
                print("비메인 스레드에서 signal.signal을 monkey-patch 하였습니다.")
            # --- v1 구동 해결책: -------------------------------------------------------------
            # engine = AsyncLLMEngine.from_engine_args(engine_args) # Original
            engine = AsyncLLMEngine.from_engine_args(engine_args) # TEST
            # v1 구동 해결책: 엔진 생성 후 원래 signal.signal으로 복원 (필요 시) -----------------
            if threading.current_thread() is not threading.main_thread():
                signal.signal = original_signal
            # --- v1 구동 해결책: -------------------------------------------------------------
            print("DEBUG: vLLM engine successfully created.") # Original
            
        except Exception as e:
            print("DEBUG: Exception during engine creation:", e)
            if "HeaderTooSmall" in str(e):
                print("DEBUG: Falling back to PyTorch weights.")
                fallback_dir = None
                for root, dirs, files in os.walk(local_model_path):
                    for file in files:
                        if (
                            "pytorch_model.bin" in file
                            and os.path.getsize(os.path.join(root, file))
                            >= MIN_WEIGHT_SIZE
                        ):
                            fallback_dir = root
                            break
                    if fallback_dir:
                        break
                if fallback_dir is None:
                    logging.error(
                        "DEBUG: No PyTorch weight file found in", local_model_path
                    )
                    raise e
                engine_args.load_format = "pt"
                engine_args.model = fallback_dir
                print("DEBUG: New EngineArgs for fallback:", engine_args)
                engine = AsyncLLMEngine.from_engine_args(engine_args)
                print("DEBUG: vLLM engine created with PyTorch fallback.")
            else:
                logging.error("DEBUG: Engine creation failed:", e)
                raise e

        engine.is_vllm = True

        print("DEBUG: Loading main LLM tokenizer with token authentication.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id,
                cache_dir=config.cache_dir,
                trust_remote_code=True,
                token=token,
                local_files_only=True  # Force loading from local cache to avoid hub requests
            )
        except Exception as e:
            print("DEBUG: Exception loading main tokenizer:", e)
            raise e
        tokenizer.model_max_length = 4024
        return engine, tokenizer, embed_model, embed_tokenizer

    else:
        print("DEBUG: vLLM is not used. Loading model via standard HF method.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id,
                cache_dir=config.cache_dir,
                trust_remote_code=True,
                token=token,
            )
        except Exception as e:
            print("DEBUG: Exception loading tokenizer:", e)
            raise e
        tokenizer.model_max_length = 4024
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                cache_dir=config.cache_dir,
                # quantization_config=bnb_config,
                trust_remote_code=True,
                token=token,
            )
        except Exception as e:
            print("DEBUG: Exception loading model:", e)
            raise e
        model.eval()
        return model, tokenizer, embed_model, embed_tokenizer

# -------------------------------------------------
# Function: load_data
# -------------------------------------------------
@time_tracker
def load_data(data_path):
    global _cached_data, _cached_data_mtime
    try:
        current_mtime = os.path.getmtime(data_path)
    except Exception as e:
        print("파일 수정 시간 확인 실패:", e)
        return None

    # 캐시가 비어있거나 파일 수정 시간이 변경된 경우 데이터 재로드
    if _cached_data is None or current_mtime != _cached_data_mtime:
        with open(data_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        # --- 디버그 함수: 벡터 포맷 검사 ---
        debug_vector_format(data)

        # 데이터 전처리 (예: 리스트 변환 및 numpy, torch 변환)
        file_names = []
        chunk_ids = []  # >>> CHANGED: Added to record each chunk's ID
        titles = []
        times = []
        vectors = []
        texts = []
        texts_short = []
        texts_vis = []
        missing_time = 0

        for file_obj in data:
            for chunk in file_obj["chunks"]:
                file_names.append(file_obj["file_name"])
                chunk_ids.append(chunk.get("chunk_id", 0))  # >>> CHANGED: Record chunk_id
                try:
                    arr = np.array(chunk["vector"])
                    vectors.append(arr)
                except Exception as e:
                    logging.warning(f"[load_data] 벡터 변환 오류: {e} → 빈 벡터로 대체")
                    vectors.append(np.zeros((1, 768), dtype=np.float32))  # 임의로 1x768 형식
                
                titles.append(chunk["title"])
                
                # 날짜 파싱
                if chunk["date"]:
                    try:
                        times.append(datetime.strptime(chunk["date"], "%Y-%m-%d"))
                    except ValueError:
                        logging.warning(f"잘못된 날짜 형식: {chunk['date']} → 기본 날짜로 대체")
                        times.append(datetime.strptime("2023-10-31", "%Y-%m-%d"))
                        missing_time += 1
                else:
                    missing_time += 1
                    times.append(datetime.strptime("2023-10-31", "%Y-%m-%d"))

                texts.append(chunk["text"])
                texts_short.append(chunk["text_short"])
                texts_vis.append(chunk["text_vis"])

        # 실제 텐서로 변환
        try:
            vectors = np.array(vectors)
            vectors = torch.from_numpy(vectors).to(torch.float32)
        except Exception as e:
            logging.error(f"[load_data] 최종 벡터 텐서 변환 오류: {str(e)}")
            # 필요 시 추가 처리

        _cached_data = {
            "file_names": file_names,
            "chunk_ids": chunk_ids,  # >>> CHANGED: Saved chunk IDs here
            "titles": titles,
            "times": times,
            "vectors": vectors,
            "texts": texts,
            "texts_short": texts_short,
            "texts_vis": texts_vis,
        }
        _cached_data_mtime = current_mtime
        print(f"Data loaded! Length: {len(titles)}, Missing times: {missing_time}")
    else:
        print("Using cached data")

    return _cached_data

# -------------------------------------------------
# Function: debug_vector_format
# -------------------------------------------------
def debug_vector_format(data):
    """
    data(List[Dict]): load_data에서 JSON으로 로드된 객체.
    각 file_obj에 대해 chunks 리스트를 순회하며 vector 형식을 디버깅 출력.
    """
    print("\n[DEBUG] ===== 벡터 형식 검사 시작 =====")
    for f_i, file_obj in enumerate(data):
        file_name = file_obj.get("file_name", f"Unknown_{f_i}")
        chunks = file_obj.get("chunks", [])
        for c_i, chunk in enumerate(chunks):
            vector_data = chunk.get("vector", None)
            if vector_data is None:
                # print(f"[DEBUG] file={file_name}, chunk_index={c_i} → vector 없음(None)")
                continue
            # 자료형, 길이, shape 등 확인
            vector_type = type(vector_data)
            # shape을 안전하게 얻기 위해 np.array 변환 시도
            try:
                arr = np.array(vector_data)
                shape = arr.shape
                # print(f"[DEBUG] file={file_name}, chunk_index={c_i} → vector_type={vector_type}, shape={shape}")
            except Exception as e:
                print(f"[DEBUG] file={file_name}, chunk_index={c_i} → vector 변환 실패: {str(e)}")
    print("[DEBUG] ===== 벡터 형식 검사 종료 =====\n")

# -------------------------------------------------
# Function: random_seed
# -------------------------------------------------
@time_tracker
def random_seed(seed):
    # Set random seed for Python's built-in random module
    random.seed(seed)

    # Set random seed for NumPy
    np.random.seed(seed)

    # Set random seed for PyTorch
    torch.manual_seed(seed)

    # Ensure the same behavior on different devices (CPU vs GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.

    # Enable deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
