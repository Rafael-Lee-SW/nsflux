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
# Loads the embedding model and the main LLM model (using vLLM if specified in the config).
@time_tracker
def load_model(config):
    # -------------------------------
    # 임베딩 모델 로드
    # -------------------------------
    embed_model = AutoModel.from_pretrained(
        config.embed_model_id, cache_dir=config.cache_dir, trust_remote_code=True
    )
    embed_tokenizer = AutoTokenizer.from_pretrained(
        config.embed_model_id, cache_dir=config.cache_dir, trust_remote_code=True
    )
    embed_model.eval()  # Set the embedding model to evaluation mode.
    embed_tokenizer.model_max_length = 4096

    # -------------------------------
    # Load the main LLM model via vLLM.
    # -------------------------------
    if config.use_vllm:
        # Set up quantization if enabled.
        if config.model.quantization_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif config.model.quantization_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None

        # Construct the local model path from the cache directory.
        # 캐시 디렉터리 내의 로컬 모델 경로 생성.
        # 참고: 컨테이너는 /home/ubuntu/huggingface를 /workspace/huggingface로 마운트합니다.
        local_model_path = os.path.join(
            config.cache_dir, "models--" + config.model_id.replace("/", "--")
        )
        local_model_path = os.path.abspath(local_model_path)
        logging.info(f"vLLM용 로컬 모델 경로: {local_model_path}")

        # config.json 파일 경로 정의 및 필요시 패치.
        config_file = os.path.join(local_model_path, "config.json")
        need_patch = False

        #### Patching is starting ####
        # If config.json is not exited, patching via under process
        if not os.path.exists(config_file):
            os.makedirs(local_model_path, exist_ok=True)
            hf_config = AutoConfig.from_pretrained(
                config.model_id, cache_dir=config.cache_dir, trust_remote_code=True
            )
            config_dict = hf_config.to_dict()
            if not config_dict.get("architectures"):
                config_dict["architectures"] = ["Gemma2ForCausalLM"]
                need_patch = True
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f)
            if need_patch:
                logging.info("패치된 config 파일 저장됨: %s", config_file)
            else:
                logging.info("config 파일 저장됨: %s", config_file)
        else:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            if not config_dict.get("architectures"):
                config_dict["architectures"] = ["Gemma2ForCausalLM"]
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f)
                logging.info("기존 config 파일 패치됨: %s", config_file)
        #### Patching is done ####

        # Recursively search for weight files in the local model directory.
        # 재귀적으로 하위 디렉터리에서 weight 파일 검색.
        weight_dir, weight_format = find_weight_directory(local_model_path)
        if weight_dir is None:
            raise RuntimeError(
                f"{local_model_path} 내에서 모델 weight 파일을 찾을 수 없습니다."
            )
        else:
            logging.info(
                f"Weight 파일이 {weight_dir}에서 발견됨. load_format: {weight_format}"
            )

        # snapshot 디렉터리에 config.json 파일이 없는 경우, 루트 config.json 복사.
        snapshot_config = os.path.join(weight_dir, "config.json")
        if not os.path.exists(snapshot_config):
            shutil.copy(config_file, snapshot_config)
            logging.info(
                "루트 config.json 파일을 snapshot 디렉터리로 복사함: %s",
                snapshot_config,
            )

        # -------------------------------
        # vLLM Engine
        # -------------------------------

        # EngineArgs 생성.
        # IMPORTANT: tokenizer 필드를 원본 모델 ID로 지정.
        engine_args = AsyncEngineArgs(
            model=weight_dir,  # weight 파일이 존재하는 디렉터리 (예: snapshot 폴더)
            tokenizer=config.model_id,
            download_dir=config.cache_dir,
            trust_remote_code=True,
            config_format="hf",
            load_format=weight_format,
        )
        # **Disable CUDA graph capture by enforcing eager mode**
        # engine_args.enforce_eager = True
        logging.info(f"EngineArgs: {engine_args}")

        # AsyncLLMEngine 생성 시도.
        try:
            engine = AsyncLLMEngine.from_engine_args(engine_args)
        except Exception as e:
            if "HeaderTooSmall" in str(e):
                logging.info(
                    "Safetensors 로드 오류 감지됨. PyTorch weight로 fallback 시도합니다."
                )
                # 재검색: pytorch_model.bin이 포함된 weight directory 재검색.
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
                    logging.error("PyTorch weight 파일도 찾을 수 없습니다.")
                    raise e
                engine_args.load_format = "pt"
                engine_args.model = fallback_dir
                logging.info(f"새로운 EngineArgs (fallback): {engine_args}")
                engine = AsyncLLMEngine.from_engine_args(engine_args)
            else:
                logging.error("엔진 로드 실패: %s", e)
                raise e

        # vLLM 엔진에는 사용자 정의 속성 추가
        engine.is_vllm = True

        # 메인 LLM 토크나이저 별도로 로드.
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, cache_dir=config.cache_dir, trust_remote_code=True
        )
        tokenizer.model_max_length = 4024
        #### Return
        return engine, tokenizer, embed_model, embed_tokenizer

    else:
        # vLLM을 사용하지 않을 경우, 기본 Hugging Face 방식으로 로드.
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, cache_dir=config.cache_dir, trust_remote_code=True
        )
        tokenizer.model_max_length = 4024
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=config.cache_dir,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
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
