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
        engine_args.max_num_seqs = vllm_conf.get("max_num_seqs", 128)
        engine_args.max_num_batched_tokens = vllm_conf.get("max_num_batched_tokens", 8192)
        # engine_args.block_size = vllm_conf.get("block_size", 128)
        engine_args.gpu_memory_utilization = vllm_conf.get("gpu_memory_utilization", 0.9)

        if vllm_conf.get("disable_custom_all_reduce", False):
            engine_args.disable_custom_all_reduce = True # For Fixing the Multi GPU problem
        
        # print("Final EngineArgs:", engine_args)
        print("EngineArgs setting be finished")

        try:
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            print("DEBUG: vLLM engine successfully created.")
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


@time_tracker
def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    file_names = []
    titles = []
    times = []
    vectors = []
    texts = []
    texts_short = []
    texts_vis = []
    tmp = 0
    for file in data:
        for chunk in file["chunks"]:
            file_names.append(file["file_name"])
            vectors.append(np.array(chunk["vector"]))
            titles.append(chunk["title"])
            if chunk["date"] != None:
                times.append(datetime.strptime(chunk["date"], "%Y-%m-%d"))
            else:
                tmp += 1
                times.append(datetime.strptime("2023-10-31", "%Y-%m-%d"))
            texts.append(chunk["text"])
            texts_short.append(chunk["text_short"])
            texts_vis.append(chunk["text_vis"])
    vectors = np.array(vectors)
    vectors = torch.from_numpy(vectors).to(torch.float32)
    data_ = {
        "file_names": file_names,
        "titles": titles,
        "times": times,
        "vectors": vectors,
        "texts": texts,
        "texts_short": texts_short,
        "texts_vis": texts_vis,
    }
    print(f"Data Loaded! Full length:{len(titles)}, Time Missing:{tmp}")
    print(f"Time Max:{max(times)}, Time Min:{min(times)}")
    return data_


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
    if type == "Retrieval":
        tmp_format = {"rsp_type": "R", "rsp_tit": "남성 내부 데이터", "rsp_data": []}
        for i, form in enumerate(qry_contents):
            tmp_format_ = {
                "rsp_tit": f"{i+1}번째 검색데이터: {form['title']} (출처:{form['file_name']})",
                "rsp_data": form["contents"],
            }
            tmp_format["rsp_data"].append(tmp_format_)
        return tmp_format

    elif type == "SQL":
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
        tmp_format = {"rsp_type": "A", "rsp_tit": "답변", "rsp_data": []}
        for i, form in enumerate(qry_contents):
            if i == 0:
                tmp_format_ = {"rsp_type": "TT", "rsp_data": form}
                tmp_format["rsp_data"].append(tmp_format_)
            elif i == 1:
                tmp_format["rsp_data"].append(form)
            else:
                None

        return tmp_format

    else:
        print("Error! Type Not supported!")
        return None


@time_tracker
def process_format_to_response(*formats):
    # Get multiple formats to tuple

    ans_format = {
        "status_code": 200,
        "result": "OK",
        "detail": "",
        "evt_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "data_list": [],
    }

    for format in formats:
        ans_format["data_list"].append(format)

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
