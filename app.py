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
# GPU 단독 사용(박상제 연구원님이랑 분기점 - 연구원님 0번 GPU, 수완 1번 GPU - 기본 안정화 세팅은 0번 GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 토크나이저 병렬 처리 명시적 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("[[TEST]]")

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
from ray_deploy import init_ray, InferenceActor, InferenceService, SSEQueueManager

# ------ checking process of the thread level
import logging
import threading

# 로깅 설정: 요청 처리 시간과 현재 스레드 이름을 기록
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [%(threadName)s] %(message)s'
)

import ray
from ray import serve
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
from data_control.data_control import data_control_bp
app.register_blueprint(data_control_bp, url_prefix="/data")

# --------------------- Non Streaming part ----------------------------
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
@app.route("/query_stream", methods=["POST"])
def query_stream():
    """
    POST 방식 SSE 스트리밍 엔드포인트.
    클라이언트가 아래 필드들을 포함한 JSON을 보내면:
      - qry_id, user_id, page_id, auth_class, qry_contents, qry_time
    auth_class는 내부적으로 'admin'으로 통일합니다.
    """
    body = request.json or {}
    # 새로운 필드 추출
    qry_id = body.get("qry_id")
    user_id = body.get("user_id")
    page_id = body.get("page_id")
    auth_class = "admin"  # 어떤 값이 와도 'admin'으로 통일
    qry_contents = body.get("qry_contents", "")
    qry_time = body.get("qry_time")  # 클라이언트 측 타임스탬프

    print(f"[DEBUG] /query_stream called with qry_id='{qry_id}', user_id='{user_id}', page_id='{page_id}', qry_contents='{qry_contents}', qry_time='{qry_time}'")
    
    # 새로운 http_query 생성 – 내부 로직에서는 page_id를 채팅방 id로 사용
    http_query = {
        "qry_id": qry_id,
        "user_id": user_id,
        "page_id": page_id if page_id else str(uuid.uuid4()),
        "auth_class": auth_class,
        "qry_contents": qry_contents,
        "qry_time": qry_time
    }
    
    # 기존 request_id 대신 page_id를 SSE queue key로 사용
    print(f"[DEBUG] Built http_query: {http_query}")
    
    # Ray Serve를 통한 streaming 호출 (변경 없음, 내부 인자는 수정된 http_query)
    response = inference_handle.process_query_stream.remote(http_query)
    obj_ref = response._to_object_ref_sync()
    chat_id = ray.get(obj_ref)  # chat_id는 page_id
    print(f"[DEBUG] streaming chat_id={chat_id}")
    
    def sse_generator():
        try:
            while True:
                # SSEQueueManager에서 토큰을 가져옴 (chat_id 사용)
                token = ray.get(sse_manager.get_token.remote(chat_id, 120))
                if token is None or token == "[[STREAM_DONE]]":
                    break
                yield f"data: {token}\n\n"
        except Exception as e:
            error_token = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_token}\n\n"
        finally:
            try:
                obj_ref = inference_handle.close_sse_queue.remote(chat_id)._to_object_ref_sync()
                ray.get(obj_ref)
            except Exception as ex:
                print(f"[DEBUG] Error closing SSE queue for {chat_id}: {str(ex)}")
            print("[DEBUG] SSE closed.")

    return Response(sse_generator(), mimetype="text/event-stream")

# --------------------- CLT Streaming part ----------------------------
@app.route("/queryToSLLM", methods=["POST"])
def query_stream_to_clt():
    """
    POST 방식 SSE 스트리밍 엔드포인트.
    클라이언트가 {"qry_id": "...", "user_id": "...", "page_id": "...", "qry_contents": "...", "qry_time": "..." }
    형태의 JSON을 보내면, 내부 Ray Serve SSE 스트림을 통해 처리한 후 지정된 response_url로 SSE 청크를 전송합니다.
    """
    # POST 요청 파라미터 파싱
    body = request.json or {}
    qry_id = body.get("qry_id", "")
    user_id = body.get("user_id", "")
    page_id = body.get("page_id", "")
    auth_class = "admin"  # 모든 요청을 'admin'으로 처리
    user_input = body.get("qry_contents", "")
    qry_time = body.get("qry_time", "")
    
    response_url = config.response_url

    print(f"[DEBUG] /queryToSLLM called with qry_id='{qry_id}', user_id='{user_id}', "
          f"page_id='{page_id}', qry_contents='{user_input}', qry_time='{qry_time}', url={response_url}")
    
    # 내부 로직에서는 page_id를 채팅방 ID(또는 request_id)로 사용합니다.
    http_query = {
        "qry_id": qry_id,
        "user_id": user_id,
        "page_id": page_id if page_id else str(uuid.uuid4()),
        "auth_class": auth_class,
        "qry_contents": user_input,
        "qry_time": qry_time,
        "response_url": response_url
    }
    print(f"[DEBUG] Built http_query={http_query}")

    # Ray Serve에 SSE 스트리밍 요청 보내기
    response = inference_handle.process_query_stream.remote(http_query)
    obj_ref = response._to_object_ref_sync()
    request_id = ray.get(obj_ref)
    print(f"[DEBUG] streaming request_id={request_id}")
    
    def sse_generator(request_id, response_url):
        token_buffer = []  # To collect tokens (for answer tokens only)
        last_sent_time = time.time()  # To track the last time data was sent
        answer_counter = 1  # 답변 업데이트 순번
        try:
            while True:
                token = ray.get(sse_manager.get_token.remote(request_id, 120))
                if token is None:
                    print("[DEBUG] 토큰이 None 반환됨. 종료합니다.")
                    break

                if isinstance(token, str):
                    token = token.strip()
                if token == "[[STREAM_DONE]]":
                    print("[DEBUG] 종료 토큰([[STREAM_DONE]]) 수신됨. 스트림 종료.")
                    break

                try:
                    token_dict = json.loads(token) if isinstance(token, str) else token
                except Exception as e:
                    print(f"[ERROR] JSON 파싱 실패: {e}. 원시 토큰: '{token}'")
                    continue

                # If token is a reference token, send it immediately
                if token_dict.get("type") == "reference":
                    print(f"[DEBUG] Reference token details: {token_dict}")
                    ref_format = process_format_to_response([token_dict], qry_id, continue_="C", update_index=answer_counter)
                    print(f"[DEBUG] Sending reference data: {json.dumps(ref_format, ensure_ascii=False, indent=2)}")
                    send_data_to_server(ref_format, response_url)
                    continue

                # Otherwise, accumulate answer tokens
                token_buffer.append(token_dict)
                current_time = time.time()
                # If 1 second has passed, flush the accumulated answer tokens
                if current_time - last_sent_time >= 1:
                    if len(token_buffer) > 0:
                        # Check if any token in the buffer signals termination.
                        final_continue = "E" if any(t.get("continue") == "E" for t in token_buffer) else "C"
                        print(f"[DEBUG] Flushing {len(token_buffer)} tokens with continue flag: {final_continue}")
                        buffer_format = process_format_to_response(token_buffer, qry_id, continue_=final_continue, update_index=answer_counter)
                        send_data_to_server(buffer_format, response_url)
                        token_buffer = []  # Reset the buffer
                        last_sent_time = current_time  # Update the last sent time
                        answer_counter += 1
                if token_dict.get("continue") == "E":
                    # Immediately flush the buffer with termination flag if needed
                    if len(token_buffer) > 0:
                        print(f"[DEBUG] Immediate flush due to termination flag in buffer (size {len(token_buffer)}).")
                        buffer_format = process_format_to_response(token_buffer, qry_id, continue_="E", update_index=answer_counter)
                        send_data_to_server(buffer_format, response_url)
                        token_buffer = []
                    break
            # After loop: if tokens remain, flush them with termination flag
            if len(token_buffer) > 0:
                print(f"[DEBUG] Final flush of remaining {len(token_buffer)} tokens with end flag.")
                buffer_format = process_format_to_response(token_buffer, qry_id, continue_="E", update_index=answer_counter)
                send_data_to_server(buffer_format, response_url)
        except Exception as e:
            print(f"[ERROR] sse_generator encountered an error: {e}")
        finally:
            try:
                obj_ref = inference_handle.close_sse_queue.remote(request_id)._to_object_ref_sync()
                ray.get(obj_ref)
            except Exception as ex:
                print(f"[DEBUG] Error closing SSE queue for {request_id}: {str(ex)}")
            print("[DEBUG] SSE closed.")

    
    # 별도의 스레드에서 SSE generator 실행
    job = threading.Thread(target=sse_generator, args=(request_id, response_url), daemon=False)
    job.start()

    # 클라이언트에는 즉시 "수신양호" 메시지를 JSON 형식으로 응답
    return Response(error_format("수신양호", 200, qry_id), content_type="application/json")

# --------------------- History & Reference part ----------------------------

# 새로 추가1: request_id로 대화 기록을 조회하는 API 엔드포인트
@app.route("/history", methods=["GET"])
def conversation_history():
    request_id = request.args.get("request_id", "")
    last_index = request.args.get("last_index")
    if not request_id:
        error_resp = error_format("request_id 파라미터가 필요합니다.", 400)
        return Response(error_resp, content_type="application/json; charset=utf-8")
    
    try:
        last_index = int(last_index) if last_index is not None else None
        response = inference_handle.get_history.remote(request_id, last_index=last_index)
        # DeploymentResponse를 ObjectRef로 변환
        obj_ref = response._to_object_ref_sync()
        history_data = ray.get(obj_ref)
        return jsonify(history_data)
    except Exception as e:
        print(f"[ERROR /history] {e}")
        error_resp = error_format(f"대화 기록 조회 오류: {str(e)}", 500)
        return Response(error_resp, content_type="application/json; charset=utf-8")

# 새로 추가2: request_id로 해당 답변의 참고자료를 볼 수 있는 API
@app.route("/reference", methods=["GET"])
def get_reference():
    request_id = request.args.get("request_id", "")
    msg_index_str = request.args.get("msg_index", "")
    if not request_id or not msg_index_str:
        error_resp = error_format("request_id와 msg_index 파라미터가 필요합니다.", 400)
        return Response(error_resp, content_type="application/json; charset=utf-8")
    
    try:
        msg_index = int(msg_index_str)
        # 먼저 history를 가져옴
        response = inference_handle.get_history.remote(request_id)
        obj_ref = response._to_object_ref_sync()
        history_data = ray.get(obj_ref)
        
        history_list = history_data.get("history", [])
        if msg_index < 0 or msg_index >= len(history_list):
            return jsonify({"error": "유효하지 않은 메시지 인덱스"}), 400
        
        message = history_list[msg_index]
        if message.get("role") != "ai":
            return jsonify({"error": "해당 메시지는 AI 응답이 아닙니다."}), 400
        
        chunk_ids = message.get("references", [])
        if not chunk_ids:
            return jsonify({"references": []})
        
        # chunk_ids에 해당하는 실제 참조 데이터 조회
        ref_response = inference_handle.get_reference_data.remote(chunk_ids)
        ref_obj_ref = ref_response._to_object_ref_sync()
        references = ray.get(ref_obj_ref)
        return jsonify({"references": references})
    except Exception as e:
        print(f"[ERROR /reference] {e}")
        error_resp = error_format(f"참조 조회 오류: {str(e)}", 500)
        return Response(error_resp, content_type="application/json; charset=utf-8")
    
@app.route("/image_query", methods=["POST"])
async def image_query():
    """
    JSON 예시:
    {
        "image_data": "base64....",
        "qry_contents": "이미지 관련 질문",
        "request_id": "some_id"
    }
    """
    try:
        http_query = request.json
        # Ray Actor(inference_handle)로 전달
        result_ref = inference_handle.image_query.remote(http_query)
        result = await result_ref
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask app 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)