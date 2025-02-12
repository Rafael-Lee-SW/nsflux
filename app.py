import os
# Setting environment variable
os.environ["TRANSFORMERS_CACHE"] = "/workspace/huggingface"
os.environ["HF_HOME"] = "/workspace/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# For the Huggingface Token setting
os.environ["HF_TOKEN"] = "/home/ubuntu/.cache/huggingface"
os.environ['MKL_THREADING_LAYER']='GNU'

from flask import Flask, request, jsonify, render_template, Response
from RAG import generate_answer, execute_rag, query_sort  # 기존에 만든 RAG 시스템 불러오기
import json
import yaml
from box import Box
from utils import load_model, load_data, random_seed, process_format_to_response, process_to_format, error_format
# Call the Ray part
import ray
from ray_setup import init_ray
from ray_utils import InferenceActor

# Configuration
with open('./config.yaml', 'r') as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)
random_seed(config.seed)

# Didn't load the model and data at first anymore - 2025-02-11. 16:19

# Load Model
# model, tokenizer, embed_model, embed_tokenizer = load_model(config)

# Load Data
# data = load_data(config.data_path)

# kwargs = {
#    "model": model,
#    "tokenizer": tokenizer,
#    "embed_model": embed_model,
#    "embed_tokenizer": embed_tokenizer,
#    "data": data,
#    "config": config,
# }

########## Ray Dashboard 8265 port ##########
init_ray()

########## Create the single  Ray Actors above  only one GPU - later on, Create more actors on serveral Multi GPU ##########
inference_actor = InferenceActor.remote(config)

########## FLASK APP ##########
app = Flask(__name__)
content_type='application/json; charset=utf-8'

# 기본 페이지를 불러오는 라우트
@app.route('/')
def index():
    return render_template('index.html')  # index.html을 렌더링

# Query Endpoint 
@app.route('/query', methods=['POST'])
async def query():
    try:
        http_query = request.json # JSON Request From Client
        # call the process_query method to do async
        future = inference_actor.process_query.remote(http_query)
        response = await future
        if isinstance(response, dict):
            response = json.dumps(response, ensure_ascii=False)
        print("APP.py - Future: ", future)
        print("APP.py - response: ", response)
        return Response(response, content_type=content_type)
    except Exception as e:
        error_resp = error_format(f"서버 처리 중 오류 발생: {str(e)}", 500)
        return Response(error_resp, content_type=content_type)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
