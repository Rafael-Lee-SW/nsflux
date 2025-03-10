# app.py
import os
# Setting environment variable
# os.environ["TRANSFORMERS_CACHE"] = "/workspace/huggingface"
os.environ["HF_HOME"] = "/workspace/huggingface"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# For the Huggingface Token setting
os.environ["HF_TOKEN_PATH"] = "/root/.cache/huggingface/token"
# Change to GNU to using OpenMP. Because this is more friendly with CUDA(NVIDIA),
# and Some library(Pytorch, Numpy, vLLM etc) use the OpenMP so that set the GNU is better.
# OpenMP: Open-Multi-Processing API
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Increase download timeout (in seconds)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
# Use the vLLM as v1 version
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_STANDBY_MEM"] = "0"
os.environ["VLLM_METRICS_LEVEL"] = "1"
os.environ["VLLM_PROFILE_MEMORY"]= "1"
# GPU 단독 사용(박상제 연구원님이랑 분기점)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU1 사용
# 토크나이저 병렬 처리 명시적 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import (
    Flask,
    request,
    Response,
    render_template,
    jsonify,
    g,
    stream_with_context,
)
import json
import yaml
from box import Box
from utils import random_seed, error_format, send_data_to_server, process_format_to_response
from datetime import datetime

# Import the Ray modules
from ray_setup import init_ray
from ray import serve
from ray_utils import InferenceActor
from ray_utils import InferenceService, SSEQueueManager

# ------ checking process of the thread level
import logging
import threading

# 로깅 설정: 요청 처리 시간과 현재 스레드 이름을 기록
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [%(threadName)s] %(message)s'
)

import ray
import uuid
import asyncio
import time

# Configuration
with open("./config.yaml", "r") as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)
random_seed(config.seed)

########## Ray Dashboard 8265 port ##########
init_ray()  # Initialize the Ray
sse_manager = SSEQueueManager.options(name="SSEQueueManager").remote()
serve.start(detached=True)

#### Ray-Actor 다중 ####
inference_service = InferenceService.options(num_replicas=config.ray.actor_count).bind(config)
serve.run(inference_service)
inference_handle = serve.get_deployment_handle("inference", app_name="default")

#### Ray-Actor 단독 ####
# inference_actor = InferenceActor.options(num_cpus=config.ray.num_cpus, num_gpus=config.ray.num_gpus).remote(config)

########## FLASK APP setting ##########
app = Flask(__name__)
content_type = "application/json; charset=utf-8"


# 기본 페이지를 불러오는 라우트
@app.route("/")
def index():
    return render_template("index.html")  # index.html을 렌더링

# Test 페이지를 불러오는 라우트
@app.route("/test")
def test_page():
    return render_template("index_test.html")

# chatroomPage 페이지를 불러오는 라우트
@app.route("/chat")
def chat_page():
    return render_template("chatroom.html")

# data 관리
from data_control import data_control_bp
app.register_blueprint(data_control_bp, url_prefix="/data")

# Query Endpoint (Non-streaming)
@app.route("/query", methods=["POST"])
async def query():
    try:
        
        # Log when the query is received
        receive_time = datetime.now().isoformat()
        print(f"[APP] Received /query request at {receive_time}")
        
        # Optionally, attach the client time if desired:
        http_query = request.json  # 클라이언트로부터 JSON 요청 수신
        
        http_query["server_receive_time"] = receive_time
        
        # Ray Serve 배포된 서비스를 통해 추론 요청 (자동으로 로드밸런싱됨)
        # result = await inference_actor.process_query.remote(http_query) # 단일
        result = await inference_handle.query.remote(http_query) # 다중
        if isinstance(result, dict):
            result = json.dumps(result, ensure_ascii=False)
        print("APP.py - 결과: ", result)
        return Response(result, content_type=content_type)
    except Exception as e:
        error_resp = error_format(f"서버 처리 중 오류 발생: {str(e)}", 500)
        return Response(error_resp, content_type=content_type)

# --------------------- Streaming part ----------------------------

# Streaming Endpoint (POST 방식 SSE) → 동기식 뷰 함수로 변경
@app.route("/query_stream", methods=["POST"])
def query_stream():
    """
    POST 방식 SSE 스트리밍 엔드포인트.
    클라이언트가 {"input": "..."} 형태의 JSON을 보내면, SSE 스타일의 청크를 반환합니다.
    """
    body = request.json or {}
    user_input = body.get("input", "")
    # request_id 파트 추가
    client_request_id = body.get("request_id")
    print(f"[DEBUG] /query_stream (POST) called with user_input='{user_input}', request_id='{client_request_id}'")
    
    http_query = {"qry_contents": user_input}
    # request_id 파트 추가
    if client_request_id:
        http_query["request_id"] = client_request_id
    print(f"[DEBUG] Built http_query={http_query}")

    # Obtain request_id from Ray
    # request_id = ray.get(inference_actor.process_query_stream.remote(http_query)) # 단일
    # ----------------------------------------------------------------------------- 다중
    response = inference_handle.process_query_stream.remote(http_query)
    obj_ref = response._to_object_ref_sync()
    request_id = ray.get(obj_ref)
    # ----------------------------------------------------------------------------- 다중
    print(f"[DEBUG] streaming request_id={request_id}")

    # def sse_generator():
    #     print("[DEBUG] sse_generator started: begin pulling partial tokens in a loop")
    #     while True:
    #         partial_text = ray.get(inference_actor.pop_sse_token.remote(request_id)) # 단일
    #         if partial_text is None:
    #             print("[DEBUG] partial_text is None => no more data => break SSE loop")
    #             break
    #         if partial_text == "[[STREAM_DONE]]":
    #             print("[DEBUG] got [[STREAM_DONE]], ending SSE loop")
    #             break
    #         yield f"data: {partial_text}\n\n"
    #     # close_sse_queue 호출
    #     ray.get(inference_actor.close_sse_queue.remote(request_id)) # 단일
    #     print("[DEBUG] SSE closed.")
    
    def sse_generator():
        try:
            while True:
                # Retrieve token from SSEQueueManager
                token = ray.get(sse_manager.get_token.remote(request_id, 120))
                if token is None or token == "[[STREAM_DONE]]":
                    break
                yield f"data: {token}\n\n"
        except Exception as e:
            error_token = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_token}\n\n"
        finally:
            # Cleanup: close the SSE queue after streaming is done
            try:
                obj_ref = inference_handle.close_sse_queue.remote(request_id)._to_object_ref_sync()
                ray.get(obj_ref)
            except Exception as ex:
                print(f"[DEBUG] Error closing SSE queue for {request_id}: {str(ex)}")
            print("[DEBUG] SSE closed.")

    return Response(sse_generator(), mimetype="text/event-stream")


# --------------------- CLT Streaming part ----------------------------

@app.route("/queryToSLLM", methods=["POST"])
def query_stream_to_clt():
    """
    POST 방식 SSE 스트리밍 엔드포인트.
    클라이언트가 {"input": "..."} 형태의 JSON을 보내면, SSE 스타일의 청크를 반환합니다.
    """
    # POST 요청 params
    body = request.json or {}
    user_input = body.get("qry_contents", "")
    query_id = body.get("qry_id", "")
    response_url = config.response_url

    print(f"[DEBUG] /query_stream (POST) called with user_input='{user_input}', ID={query_id}, url={response_url}")
    http_query = {"qry_contents": user_input}
    print(f"[DEBUG] Built http_query={http_query}")

    # Obtain request_id from Ray
    response = inference_handle.process_query_stream.remote(http_query, query_id=query_id, response_url=response_url)
    obj_ref = response._to_object_ref_sync()
    request_id = ray.get(obj_ref)

    print(f"[DEBUG] streaming request_id={request_id}")
    
    def sse_generator(request_id, response_url):
        try:
            token_buffer = []  # To collect tokens
            last_sent_time = time.time()  # To track the last time data was sent

            while True:
                # Retrieve token from SSEQueueManager
                token = ray.get(sse_manager.get_token.remote(request_id, 120))
                token_dict = json.loads(token) if isinstance(token, str) else token
                
                token_buffer.append(token_dict)  # Collect token
                
                current_time = time.time()

                # If 1 second has passed, send the accumulated tokens
                if current_time - last_sent_time >= 1:
                    # Send the accumulated tokens
                    buffer_format = process_format_to_response(token_buffer, request_id)
                    send_data_to_server(buffer_format, response_url)
                    token_buffer = []  # Reset the buffer
                    last_sent_time = current_time  # Update the last sent time
                
                # If "continue" is "E", send the accumulated tokens with END signal
                elif token_dict.get("continue") == "E":
                    # Send the accumulated tokens --- EXCEPT LAST END TOKEN
                    buffer_format = process_format_to_response(token_buffer[:-1], request_id, continue_="E")
                    send_data_to_server(buffer_format, response_url)
                    token_buffer = []  # Reset the buffer
                    last_sent_time = current_time  # Update the last sent time

                # If the "continue" key indicates to stop, break the loop
                if token_dict.get("continue") == "E":
                    break

        except Exception as e:
            # error_token = json.dumps({"type": "error", "message": str(e)})
            # yield f"data: {error_token}\n\n"
            print(e)

        finally:
            # Cleanup: close the SSE queue after streaming is done
            try:
                obj_ref = inference_handle.close_sse_queue.remote(request_id)._to_object_ref_sync()
                ray.get(obj_ref)
            except Exception as ex:
                print(f"[DEBUG] Error closing SSE queue for {request_id}: {str(ex)}")
            print("[DEBUG] SSE closed.")
    
    job = threading.Thread(target=sse_generator, args=(request_id, response_url), daemon=False)
    job.start()

    return Response(error_format("수신양호", 200), content_type="application/json")


# Flask app 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
