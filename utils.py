# utils.py
import json
import numpy as np
import torch
import random
from datetime import datetime, timedelta
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

import os

# vLLM 관련 임포트
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# Tracking
from tracking import time_tracker

# Logging
import logging


@time_tracker
def load_model(config):
    ### 임베딩 모델 로드 (변경 없음) ###
    embed_model = AutoModel.from_pretrained(
        config.embed_model_id, cache_dir=config.cache_dir
    )
    embed_tokenizer = AutoTokenizer.from_pretrained(
        config.embed_model_id, cache_dir=config.cache_dir
    )
    embed_model.eval()
    embed_tokenizer.model_max_length = 4096

    ### LLM 모델 로드 ###
    if config.use_vllm:
        # vLLM 엔진을 사용하여 모델 로드
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

        # 로컬 모델 경로 구성: Hugging Face 캐시 폴더 내에 다운로드된 모델 경로
        local_model_path = os.path.join(
            config.cache_dir, "models--" + config.model_id.replace("/", "--")
        )
        local_model_path = os.path.abspath(local_model_path)  # 절대 경로로 변환
        logging.info(f"Using local model path for vLLM: {local_model_path}")

        # EngineArgs 객체 생성 시 로컬 모델 경로 사용
        engine_args = AsyncEngineArgs(
            model=local_model_path,
            download_dir=config.cache_dir,
            trust_remote_code=True,  # 필요 시 True로 설정 (Gemma-2와 같은 커스텀 모델)
            # 추가적으로 필요한 인자(예: trust_remote_code) 필요 시 설정
        )
        logging.info(f"EngineArgs: {engine_args}")

        # vLLM 엔진 생성
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logging.info(f"After Engine creation: {engine}")
        print(f"After Engine : ", engine)

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, cache_dir=config.cache_dir
        )
        tokenizer.model_max_length = 4024

        return engine, tokenizer, embed_model, embed_tokenizer
    else:
        # 기존 Hugging Face 방식
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, cache_dir=config.cache_dir
        )
        tokenizer.model_max_length = 4024
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

        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=config.cache_dir,
            quantization_config=bnb_config,
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
