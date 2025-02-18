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
from ray_utils import InferenceActor

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

########## Create the single Ray Actors above only one GPU ##########
########## later on, Create more actors on serveral Multi GPU ##########
inference_actor = InferenceActor.remote(config)

########## FLASK APP setting ##########
app = Flask(__name__)
content_type='application/json; charset=utf-8'

# 기본 페이지를 불러오는 라우트
@app.route('/')
def index():
    return render_template('index.html') # index.html을 렌더링

# Test 페이지를 불러오는 라우트
# @app.route('/test')
# def test_page():
#     return render_template('index_test.html') # index.html을 렌더링

# Test 페이지를 불러오는 라우트
@app.route('/stream')
def test_page():
    return render_template('index_test_streaming.html') # index.html을 렌더링

# Query Endpoint
@app.route('/query', methods=['POST'])
async def query():
    try:
        http_query = request.json # JSON Request From Client
        future = inference_actor.process_query.remote(http_query) # Inference Process is starting at here
        response = await future # Wait a Result of Inference
        # If response is dictionary type, translate to JSON
        if isinstance(response, dict):
            response = json.dumps(response, ensure_ascii=False)
        # Print the Result
        print("APP.py - Future: ", future)
        print("APP.py - response: ", response)
        return Response(response, content_type=content_type)
    except Exception as e:
        error_resp = error_format(f"서버 처리 중 오류 발생: {str(e)}", 500)
        return Response(error_resp, content_type=content_type)

# Flask app 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

# --------------------- Streaming part ----------------------------

@app.route('/query_stream', methods=['POST'])
def query_stream():
    """
    SSE streaming endpoint.
    Client sends JSON { "qry_contents": "..." }
    We:
      1) ask the actor for a request_id via process_query_stream.remote(http_query).
      2) yield partial tokens in SSE until done.
    """
    http_query = request.json or {}
    # call actor
    request_id_future = inference_actor.process_query_stream.remote(http_query)
    request_id = ray.get(request_id_future)
    print("Assigned request_id:", request_id)

    @stream_with_context
    def sse_generator():
        # We'll pull partial tokens in a loop
        while True:
            # We call pop_sse_token in an async manner. But Flask route is sync.
            # So we can do an async->sync bridge:
            partial_text = ray.get(inference_actor.pop_sse_token.remote(request_id))
            if partial_text is None:
                # Means no data or the queue is done. 
                # We might sleep or break
                break

            if partial_text == "[[STREAM_DONE]]":
                # End of the generation
                break

            # yield in SSE format
            yield f"data: {partial_text}\n\n"

        # Now we can close the queue
        ray.get(inference_actor.close_sse_queue.remote(request_id))

    return Response(sse_generator(), mimetype='text/event-stream')

# --------------------- Streaming part ----------------------------