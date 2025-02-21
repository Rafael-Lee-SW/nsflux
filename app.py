# app.py
import os
# Setting environment variable
os.environ["TRANSFORMERS_CACHE"] = "/workspace/huggingface"
os.environ["HF_HOME"] = "/workspace/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# For the Huggingface Token setting
os.environ["HF_TOKEN_PATH"] = "/root/.cache/huggingface/token"
# Change to GNU to using OpenMP. Because this is more friendly with CUDA(NVIDIA),
# and Some library(Pytorch, Numpy, vLLM etc) use the OpenMP so that set the GNU is better.
# OpenMP: Open-Multi-Processing API
os.environ['MKL_THREADING_LAYER']='GNU'
# Increase download timeout (in seconds)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

from flask import Flask, request, Response, render_template, jsonify, stream_with_context
import json
import yaml
from box import Box
from utils import random_seed, error_format
# Import the Ray modules
from ray_setup import init_ray
from ray import serve
from ray_utils import InferenceActor
from ray_utils import InferenceService

# --------------------- Streaming part ----------------------------
import ray
import uuid
import asyncio
# --------------------- Streaming part ----------------------------

# Configuration
with open('./config.yaml', 'r') as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)
random_seed(config.seed)

########## Ray Dashboard 8265 port ##########
init_ray() # Initialize the Ray
serve.start(detached=True)

# Ray Serve 배포(Deployment) - InferenceService는 ray_utils.py에 정의되어 있음.
# config.ray.actor_count에 따라 복제본(replica) 수를 결정합니다.
inference_service = InferenceService.bind(config)
serve.run(inference_service)

# 배포된 서비스 핸들 가져오기
inference_handle = serve.get_deployment_handle("inference", app_name="default")

########## FLASK APP setting ##########
app = Flask(__name__)
content_type='application/json; charset=utf-8'

# 기본 페이지를 불러오는 라우트
@app.route('/')
def index():
    return render_template('index.html') # index.html을 렌더링

# Test 페이지를 불러오는 라우트
@app.route('/test')
def test_page():
    return render_template('index_test_streaming.html') # index.html을 렌더링
#     return render_template('index_test.html') # index.html을 렌더링

# Query Endpoint (Non-streaming)
@app.route('/query', methods=['POST'])
async def query():
    try:
        http_query = request.json  # 클라이언트로부터 JSON 요청 수신
        # Ray Serve 배포된 서비스를 통해 추론 요청 (자동으로 로드밸런싱됨)
        result = await inference_handle.query.remote(http_query)
        if isinstance(result, dict):
            result = json.dumps(result, ensure_ascii=False)
        print("APP.py - 결과: ", result)
        return Response(result, content_type=content_type)
    except Exception as e:
        error_resp = error_format(f"서버 처리 중 오류 발생: {str(e)}", 500)
        return Response(error_resp, content_type=content_type)

# Query Streaming Endpoint (SSE)
@app.route('/query_stream', methods=['GET'])
def query_stream():
    """
    SSE 스트리밍 엔드포인트.
    클라이언트로부터 입력을 받고, Ray Serve를 통해 streaming request_id를 받아
    반복적으로 partial token을 클라이언트로 전송합니다.
    """
    user_input = request.args.get('input', '')
    print(f"[DEBUG] /query_stream 호출, 입력: '{user_input}'")
    http_query = {"qry_contents": user_input}
    print(f"[DEBUG] http_query 구성: {http_query}")
    
    # Ray Serve 배포된 서비스를 통해 streaming 요청 처리
    request_id = ray.get(inference_handle.process_query_stream.remote(http_query))
    print(f"[DEBUG] 요청 ID 반환: {request_id}")

    @stream_with_context
    def sse_generator():
        print("[DEBUG] sse_generator 시작: partial token 수신 대기")
        while True:
            token = ray.get(inference_handle.pop_sse_token.remote(request_id))
            print(f"[DEBUG] 수신된 token: {token}")
            if token is None or token == "[[STREAM_DONE]]":
                break
            yield f"data: {token}\n\n"
        print("[DEBUG] SSE 종료: close_sse_queue 호출")
        ray.get(inference_handle.close_sse_queue.remote(request_id))
    return Response(sse_generator(), mimetype='text/event-stream')

# --------------------- Streaming part ----------------------------

# Flask app 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)