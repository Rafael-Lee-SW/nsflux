# utils.py
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

import os

# 전역 캐시 변수 - 데이터의 변화를 감지하기 위한
_cached_data = None
_cached_data_mtime = 0

# Import vLLM utilities
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# Define the minimum valid file size (e.g., 10MB)
MIN_WEIGHT_SIZE = 10 * 1024 * 1024

# For tracking execution time of functions
from tracking import time_tracker

# Logging
import logging

logging.basicConfig(level=logging.DEBUG)

# -------------------------------------------------
# Function: find_weight_directory
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
            token=token,  # using 'token' parameter
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
    embed_tokenizer.model_max_length = 4096

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
            config_dict = hf_config.to_dict()
            if not config_dict.get("architectures"):
                config_dict["architectures"] = ["Gemma2ForCausalLM"]
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f)
        else:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            if not config_dict.get("architectures"):
                config_dict["architectures"] = ["Gemma2ForCausalLM"]
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
        
        engine_args.enable_prefix_caching = True
        engine_args.scheduler_delay_factor = vllm_conf.get("scheduler_delay_factor", 0.1)
        engine_args.enable_chunked_prefill = True
        engine_args.tensor_parallel_size = vllm_conf.get("tensor_parallel_size", 1) # Using Multi-GPU at once.
        # engine_args.max_num_seqs = vllm_conf.get("max_num_seqs")
        engine_args.max_num_batched_tokens = vllm_conf.get("max_num_batched_tokens", 8192)
        # engine_args.block_size = vllm_conf.get("block_size", 128)
        engine_args.gpu_memory_utilization = vllm_conf.get("gpu_memory_utilization")
        
        if vllm_conf.get("disable_custom_all_reduce", False):
            engine_args.disable_custom_all_reduce = True # For Fixing the Multi GPU problem
        
        # engine_args.enable_memory_defrag = True # v1 새로운 기능
        # engine_args.max_model_len = vllm_conf.get("max_model_len") # Context Length
        
        # # ★★ 추가: 슬라이딩 윈도우 비활성화 옵션 적용 ★★
        # if vllm_conf.get("disable_sliding_window", False):
        #     # cascade attention에서는 슬라이딩 윈도우가 (-1, -1)이어야 함
        #     engine_args.sliding_window = (-1, -1)
        #     print("Sliding window disabled: engine_args.sliding_window set to (-1, -1)")
        
        # print("Final EngineArgs:", engine_args)
        print("EngineArgs setting be finished")
        
        #         # ── 여기서 unified_attention 호출 추적을 위한 monkey-patch ──
        # try:
        #     if hasattr(torch.ops.vllm, "unified_attention_with_output"):
        #         orig_unified_attention = torch.ops.vllm.unified_attention_with_output
        #         def tracking_unified_attention(*args, **kwargs):
        #             logging.info("Called unified_attention_with_output with args: %s, kwargs: %s", args, kwargs)
        #             return orig_unified_attention(*args, **kwargs)
        #         torch.ops.vllm.unified_attention_with_output = tracking_unified_attention
        #         logging.info("Monkey-patched unified_attention_with_output for tracking.")
        # except Exception as e:
        #     logging.warning("Failed to monkey-patch unified_attention_with_output: %s", e)
        # # ── 끝 ──

        try:
            # --- v1 구동 해결책: 현재 스레드가 메인 스레드가 아니면 signal 함수를 임시 패치 ---
            import threading, signal
            if threading.current_thread() is not threading.main_thread():
                original_signal = signal.signal
                signal.signal = lambda s, h: None  # signal 설정 무시
                print("비메인 스레드에서 signal.signal을 monkey-patch 하였습니다.")
            # --- v1 구동 해결책: ------------------------------------------------------ ---
            engine = AsyncLLMEngine.from_engine_args(engine_args) # Original
            # v1 구동 해결책: 엔진 생성 후 원래 signal.signal으로 복원 (필요 시) ----------------- ---
            if threading.current_thread() is not threading.main_thread():
                signal.signal = original_signal
            # --- v1 구동 해결책: ------------------------------------------------------ ---
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
                quantization_config=bnb_config,
                trust_remote_code=True,
                token=token,
            )
        except Exception as e:
            print("DEBUG: Exception loading model:", e)
            raise e
        model.eval()
        return model, tokenizer, embed_model, embed_tokenizer

# @time_tracker
# def load_data(data_path):
#     global _cached_data, _cached_data_mtime
#     try:
#         current_mtime = os.path.getmtime(data_path)
#     except Exception as e:
#         print("파일 수정 시간 확인 실패:", e)
#         return None

#     # 캐시가 비어있거나 파일 수정 시간이 변경된 경우 데이터 재로드
#     if _cached_data is None or current_mtime != _cached_data_mtime:
#         with open(data_path, "r", encoding="utf-8") as json_file:
#             data = json.load(json_file)
#         # 데이터 전처리 (예: 리스트 변환 및 numpy, torch 변환)
#         file_names = []
#         titles = []
#         times = []
#         vectors = []
#         texts = []
#         texts_short = []
#         texts_vis = []
#         missing_time = 0
#         for file in data:
#             for chunk in file["chunks"]:
#                 file_names.append(file["file_name"])
#                 vectors.append(np.array(chunk["vector"]))
#                 titles.append(chunk["title"])
#                 if chunk["date"]:
#                     times.append(datetime.strptime(chunk["date"], "%Y-%m-%d"))
#                 else:
#                     missing_time += 1
#                     times.append(datetime.strptime("2023-10-31", "%Y-%m-%d"))
#                 texts.append(chunk["text"])
#                 texts_short.append(chunk["text_short"])
#                 texts_vis.append(chunk["text_vis"])
#         vectors = np.array(vectors)
#         vectors = torch.from_numpy(vectors).to(torch.float32)
#         _cached_data = {
#             "file_names": file_names,
#             "titles": titles,
#             "times": times,
#             "vectors": vectors,
#             "texts": texts,
#             "texts_short": texts_short,
#             "texts_vis": texts_vis,
#         }
#         _cached_data_mtime = current_mtime
#         print(f"Data loaded! Length: {len(titles)}, Missing times: {missing_time}")
#     else:
#         print("Using cached data")
#     return _cached_data

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


@time_tracker
def process_to_format(qry_contents, type):
    # 여기서 RAG 시스템을 호출하거나 답변을 생성하도록 구현하세요.
    # 예제 응답 형식
    ### rsp_type : RA(Retrieval All), RT(Retrieval Text), RB(Retrieval taBle), AT(Answer Text), AB(Answer taBle) ###
    print("[SOOWAN] process_to_format 진입")
    if type == "Retrieval":
        print("[SOOWAN] 타입 : 리트리버")
        tmp_format = {"rsp_type": "R", "rsp_tit": "남성 내부 데이터", "rsp_data": []}
        for i, form in enumerate(qry_contents):
            tmp_format_ = {
                "rsp_tit": f"{i+1}번째 검색데이터: {form['title']} (출처:{form['file_name']})",
                "rsp_data": form["contents"],
            }
            tmp_format["rsp_data"].append(tmp_format_)
        return tmp_format

    elif type == "SQL":
        print("[SOOWAN] 타입 : SQL")
        tmp_format = {
            "rsp_type": "R",
            "rsp_tit": "남성 내부 데이터",
            "rsp_data": [{"rsp_tit": "SQL Query 결과표", "rsp_data": []}],
        }
        tmp_format_sql = {
            "rsp_type": "TB",
            "rsp_tit": qry_contents[0]["title"],
            "rsp_data": qry_contents[0]["data"],
        }
        tmp_format_chart = {
            "rsp_type": "CT",
            "rsp_tit": qry_contents[1]["title"],
            "rsp_data": {"chart_tp": "BAR", "chart_data": qry_contents[1]["data"]},
        }
        tmp_format["rsp_data"][0]["rsp_data"].append(tmp_format_sql)
        # tmp_format['rsp_data'].append(tmp_format_chart)
        return tmp_format, tmp_format_chart

    elif type == "Answer":
        print("[SOOWAN] 타입 : 대답")
        tmp_format = {"rsp_type": "A", "rsp_tit": "답변", "rsp_data": []}
        # for i, form in enumerate(qry_contents):
            # if i == 0:
        tmp_format_ = {"rsp_type": "TT", "rsp_data": qry_contents}
        tmp_format["rsp_data"].append(tmp_format_)
            # elif i == 1:
            #     tmp_format["rsp_data"].append(form)
            # else:
            #     None

        return tmp_format

    else:
        print("Error! Type Not supported!")
        return None


@time_tracker
def process_format_to_response(formats, qry_id, continue_="C"):
    # Get multiple formats to tuple

    ans_format = {
        "status_code": 200,
        "result": "OK",
        "detail": "",
        "continue":continue_,
        "qry_id": qry_id,
        "rsp_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "data_list": [],
    }

    for format in formats:
        ans_format["data_list"].append(format)

    # return json.dumps(ans_format, ensure_ascii=False)
    return ans_format


@time_tracker
def error_format(message, status):
    ans_format = {
        "status_code": status,
        "result": message,
        "detail": "",
        "evt_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
    }
    return json.dumps(ans_format)

@time_tracker
def send_data_to_server(data, url):

    headers = {
        "Content-Type": "application/json; charset=utf-8"
    }

    try:
        # 다른 서버로 데이터를 전송 (POST 요청)
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            print(f"Data sent successfully: {data}")
        else:
            print(f"Failed to send data: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending data: {e}")

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
    