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
    for file in data:
        for chunk in file['chunks']:
            file_names.append(file['file_name'])
            vectors.append(np.array(chunk['vector']))
            titles.append(chunk['title'])
            if chunk["date"] != "":
                times.append(datetime.strptime(chunk["date"],"%Y-%m-%d"))
            else:
                times.append("all")
            texts.append(chunk['text'])
            texts_short.append(chunk['text_short'])
    vectors = np.array(vectors)
    vectors = torch.from_numpy(vectors).to(torch.float32)
    data_ = {'file_names':file_names,
            'titles':titles,
            'times':times,
            'vectors':vectors,
            'texts':texts,
            'texts_short':texts_short}
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