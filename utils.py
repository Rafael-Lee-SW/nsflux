import json
import numpy as np
import torch
import random
from datetime import datetime, timedelta
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_model(config):
    ### 임베딩 모델 ###
    embed_model = AutoModel.from_pretrained(config.embed_model_id, cache_dir = config.cache_dir)
    embed_tokenizer = AutoTokenizer.from_pretrained(config.embed_model_id, cache_dir = config.cache_dir)
    embed_model.eval()

    ### LLM 모델 ###
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, cache_dir = config.cache_dir)
    if config.model.quantization == True:
        # bnb_config = BitsAndBytesConfig(
        #         load_in_4bit=True
        #         )
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
                )

    else:
        bnb_config = None
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir = config.cache_dir,
        quantization_config = bnb_config
        )
    model.eval()
    return model,tokenizer,embed_model,embed_tokenizer


def load_data(data_path):
    with open(data_path, 'r') as json_file:
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
        for chunk in file['chunks']:
            file_names.append(file['file_name'])
            vectors.append(np.array(chunk['vector']))
            titles.append(chunk['title'])
            if chunk["date"] != None:
                times.append(datetime.strptime(chunk["date"],"%Y-%m-%d"))
            else:
                tmp += 1
                times.append("all")
            texts.append(chunk['text'])
            texts_short.append(chunk['text_short'])
            texts_vis.append(chunk['text_vis'])
    vectors = np.array(vectors)
    vectors = torch.from_numpy(vectors).to(torch.float32)
    data_ = {'file_names':file_names,
            'titles':titles,
            'times':times,
            'vectors':vectors,
            'texts':texts,
            'texts_short':texts_short,
            'texts_vis':texts_vis}
    print(f"Data Loaded! Full length:{len(titles)}, Time Missing:{tmp}")
    return data_

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

def process_to_format(qry_contents, type):
    # 여기서 RAG 시스템을 호출하거나 답변을 생성하도록 구현하세요.
    # 예제 응답 형식
    ### rsp_type : RA(Retrieval All), RT(Retrieval Text), RB(Retrieval taBle), AT(Answer Text), AB(Answer taBle) ###
    if type == "Retrieval":
        tmp_format = {
            "rsp_type": "R", "rsp_tit": "남성 내부 데이터", "rsp_data": []
        }
        for i, form in enumerate(qry_contents):
            tmp_format_ = {
                "rsp_tit": f"{i+1}번째 검색데이터", "rsp_data": form
            }
            tmp_format['rsp_data'].append(tmp_format_)
        return tmp_format
    
    elif type == "SQL":
        tmp_format = {
            "rsp_type": "R", "rsp_tit": "남성 내부 데이터", "rsp_data": []
        }
        for i,form in enumerate(qry_contents):
            tmp_format_ = {
                "rsp_tit": "SQL 추출 내부 정형데이터", "rsp_data":[
                    {
                        "rsp_type":"TT", "rsp_data":form
                    }
                ]
            }
            tmp_format['rsp_data'].append(tmp_format_)
        return tmp_format

    elif type == "Answer":
        tmp_format = {
            "rsp_type": "A", "rsp_tit": "답변", "rsp_data": []
        }
        for i,form in enumerate(qry_contents):
            tmp_format_ = {
                "rsp_type": "TT", "rsp_data": form
            }
            tmp_format['rsp_data'].append(tmp_format_)
        return tmp_format

    else:
        print("Error! Type Not supported!")
        return None

def process_format_to_response(*formats):
    # Get multiple formats to tuple

    ans_format = {
        "status_code": 200,
        "result": "OK",
        "detail": "",
        "evt_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "data_list": []
    }

    for format in formats:
        ans_format['data_list'].append(format)

    return ans_format