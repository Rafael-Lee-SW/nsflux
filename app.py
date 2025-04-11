# app.py

########## Setting the Thread to main ##########
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

# Setting environment variable
import os

# --------- Huggingface setting ---------
# For the Huggingface Token setting
os.environ["HF_HOME"] = "/workspace/huggingface"
os.environ["HF_TOKEN_PATH"] = "/root/.cache/huggingface/token"
# Increase download timeout (in seconds)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
# --------- OpenMP: Open-Multi-Processing API setting ---------
# Change to GNU to using OpenMP. Because this is more friendly with CUDA(NVIDIA), and Some library(Pytorch, Numpy, vLLM etc) use the OpenMP so that set the GNU is better.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# --------- vLLM ---------
# Use the vLLM as v1 version
os.environ["VLLM_USE_V1"] = "1"

# ---------------------- Weird configuration of them ----------------------
# os.environ["VLLM_STANDBY_MEM"] = "0"
# Metrics logging configuration
# os.environ["VLLM_METRICS_LEVEL"] = "2"  # Increase to max level
# os.environ["VLLM_LOG_STATS_INTERVAL"] = "5"  # Report every 5 seconds
# os.environ["VLLM_SHOW_PROGRESS"] = "1"  # Show progress bars
# os.environ["VLLM_LOG_LEVEL"] = "DEBUG"  # Set log level to DEBUG
# os.environ["VLLM_PROFILE_MEMORY"]= "1"
# ------------------------------------------------------------------------
# TEST env
# os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"         # 로그 레벨을 DEBUG로 변경 - configuration 단계에서만 다 보여주고 engine 구동 중에 토큰 관련한 내용은 여전히 안보임
# os.environ["VLLM_TRACE_FUNCTION"] = "1"            # 함수 호출 추적 활성화
# os.environ["VERBOSE"] = "1"                        # 설치 및 디버깅 로그 활성화
os.environ["VLLM_CONFIGURE_LOGGING"] = "1"

# GPU 단독 사용(박상제 연구원님이랑 분기점 - 연구원님 0번 GPU, 수완 1번 GPU - 2GPU 시에 해당 설정을 없애주어야 함)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 토크나이저 병렬 처리 명시적 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# VLLM FLASH ATTENTION SETTING
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
# VLLM
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# --------- Ray ---------
os.environ["RAY_DEDUP_LOGS"] = "0"

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
from datetime import datetime
from flask_cors import CORS

# Import the Ray modules
from ray_deploy import init_ray, InferenceService, SSEQueueManager

# Import utils
from utils import (
    random_seed,
    error_format,
    send_data_to_server,
    process_format_to_response,
)

# ------ checking process of the thread level
import logging
import threading

# Metrics logging configuration - 여기에 모든 로깅 설정 통합
import logging
from logging.handlers import RotatingFileHandler
import sys

# 기본은 INFO
logging.basicConfig(level=logging.INFO)

# 글로벌 로깅 설정
# app.py 로깅 설정 수정
def setup_logging():
    """
    통합 로깅 설정 - 중복 로그 방지
    """
    # 로그 포맷 설정
    log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] %(message)s"

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.isEnabledFor(logging.INFO)

    # 기존 핸들러 제거 (중복 방지)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    # 성능 전용 로거 설정 - 중복 로그 방지를 위한 추가 확인
    perf_logger = logging.getLogger("performance")
    # 기존 핸들러 모두 제거
    for handler in perf_logger.handlers[:]:
        perf_logger.removeHandler(handler)

    # 변경: propagate를 True로 설정하거나 독립 핸들러 추가
    perf_logger.propagate = False  # 루트 로거로 전파 허용
    perf_logger.setLevel(logging.INFO)

    # 성능 로거 핸들러 추가
    perf_handler = logging.StreamHandler(sys.stdout)
    perf_handler.setFormatter(logging.Formatter(log_format))
    perf_logger.addHandler(perf_handler)

    # 파일 로깅 설정 (폴더 생성 후)
    try:
        os.makedirs("logs", exist_ok=True)

        # 일반 로그 파일 핸들러
        file_handler = RotatingFileHandler(
            "logs/server.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

        # 성능 전용 로그 파일
        perf_file_handler = RotatingFileHandler(
            "logs/performance.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        perf_file_handler.setFormatter(logging.Formatter(log_format))
        perf_logger.addHandler(perf_file_handler)

    except (FileNotFoundError, PermissionError) as e:
        logging.warning(
            f"로그 파일을 생성할 수 없습니다: {e}. 콘솔 로깅만 활성화됩니다."
        )

    # Ray 로거와 vLLM 로거 제어
    ray_logger = logging.getLogger("ray")
    ray_logger.setLevel(logging.WARNING)  # Ray 로그 축소

    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(logging.INFO)  # vLLM 로그는 INFO 유지

    # 스레드 이름으로 로깅되는 것 방지
    thread_logger = logging.getLogger("Thread")
    thread_logger.setLevel(logging.WARNING)

    # 로깅 설정 확인
    logging.info("로깅 시스템 초기화 완료 - 중복 로그 방지")


# 로깅 설정 적용
setup_logging()

import ray
from ray import serve
import uuid
import asyncio
import time

# Configuration
from config import config

random_seed(config.seed)

# RAG 서비스 URL 설정이 없는 경우 기본값 설정
if not hasattr(config, "rag_service_url"):
    config.rag_service_url = "http://localhost:8000"
    logging.info(
        f"RAG 서비스 URL이 설정되지 않았습니다. 기본값 '{config.rag_service_url}'을 사용합니다."
    )
else:
    logging.info(f"RAG 서비스 URL: {config.rag_service_url}")

########## Ray Dashboard 8265 port ##########
init_ray()  # Initialize the Ray

sse_manager = SSEQueueManager.options(name="SSEQueueManager").remote()
serve.start(detached=True)

#### Ray-Actor 다중 ####
inference_service = InferenceService.options(num_replicas=config.ray.actor_count).bind(
    config
)
serve.run(inference_service)
inference_handle = serve.get_deployment_handle("inference", app_name="default")

#### Ray-Actor 단독 ####
# inference_actor = InferenceActor.options(num_cpus=config.ray.num_cpus, num_gpus=config.ray.num_gpus).remote(config)

########## FLASK APP setting ##########
app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": ["http://localhost:3000"],  # 프론트엔드 개발 서버 주소
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
        }
    },
)
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
        result = await inference_handle.query.remote(http_query)  # 다중
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
    # 요청에서 바로 body 받아오기
    body = request.json or {}

    # 새로운 필드 추출
    qry_id = body.get("qry_id")
    user_id = body.get("user_id")
    page_id = body.get("page_id")
    auth_class = "admin"  # 어떤 값이 와도 'admin'으로 통일
    qry_contents = body.get("qry_contents", "")
    qry_time = body.get("qry_time")  # 클라이언트 측 타임스탬프
    image_data = body.get("image_data")  # Image Data
    # RAG 여부
    use_rag = body.get("rag")
    print(f"[DEBUG] RAG using = ", use_rag)

    print(
        f"[DEBUG] /query_stream called with qry_id='{qry_id}', user_id='{user_id}', page_id='{page_id}', qry_contents='{qry_contents}', qry_time='{qry_time}'"
    )

    # SSE 큐 생성
    chat_id = body.get("page_id") or str(uuid.uuid4())  # 뒤에서 다시 초기화됨
    ray.get(sse_manager.create_queue.remote(chat_id))

    # 새로운 http_query 생성 – 내부 로직에서는 page_id를 채팅방 id로 사용
    http_query = {
        "qry_id": qry_id,
        "user_id": user_id,
        "page_id": page_id if page_id else str(uuid.uuid4()),
        "auth_class": auth_class,
        "qry_contents": qry_contents,
        "qry_time": qry_time,
        "use_rag": use_rag,
    }

    # image_data가 존재하면 http_query에 추가하고, 길이(또는 타입)만 간략하게 출력
    if image_data is not None:
        http_query["image_data"] = image_data
        # image_data가 문자열이나 시퀀스 타입이면 길이를, 아니면 타입을 출력합니다.
        if hasattr(image_data, "__len__"):
            print(f"[DEBUG] image_data received: length={len(image_data)}")
        else:
            print(f"[DEBUG] image_data received: type={type(image_data)}")

    # http_query 전체를 출력할 때 image_data 내용은 생략(요약 정보만 출력)
    http_query_print = http_query.copy()
    if "image_data" in http_query_print:
        http_query_print["image_data"] = "<omitted>"
    print(f"[DEBUG] Built http_query: {http_query_print}")

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
                ray.get(sse_manager.delete_queue.remote(chat_id))
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

    print(
        f"[DEBUG] /queryToSLLM called with qry_id='{qry_id}', user_id='{user_id}', "
        f"page_id='{page_id}', qry_contents='{user_input}', qry_time='{qry_time}', url={response_url}"
    )

    # 내부 로직에서는 page_id를 채팅방 ID(또는 request_id)로 사용합니다.
    http_query = {
        "qry_id": qry_id,
        "user_id": user_id,
        "page_id": page_id if page_id else str(uuid.uuid4()),
        "auth_class": auth_class,
        "qry_contents": user_input,
        "qry_time": qry_time,
        "response_url": response_url,
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
                    ref_format = process_format_to_response(
                        [token_dict], qry_id, continue_="C", update_index=answer_counter
                    )
                    print(
                        f"[DEBUG] Sending reference data: {json.dumps(ref_format, ensure_ascii=False, indent=2)}"
                    )
                    send_data_to_server(ref_format, response_url)
                    continue

                # Otherwise, accumulate answer tokens
                token_buffer.append(token_dict)
                current_time = time.time()
                # If 1 second has passed, flush the accumulated answer tokens
                if current_time - last_sent_time >= 1:
                    if len(token_buffer) > 0:
                        # Check if any token in the buffer signals termination.
                        final_continue = (
                            "E"
                            if any(t.get("continue") == "E" for t in token_buffer)
                            else "C"
                        )
                        print(
                            f"[DEBUG] Flushing {len(token_buffer)} tokens with continue flag: {final_continue}"
                        )
                        buffer_format = process_format_to_response(
                            token_buffer,
                            qry_id,
                            continue_=final_continue,
                            update_index=answer_counter,
                        )
                        send_data_to_server(buffer_format, response_url)
                        token_buffer = []  # Reset the buffer
                        last_sent_time = current_time  # Update the last sent time
                        answer_counter += 1
                if token_dict.get("continue") == "E":
                    # Immediately flush the buffer with termination flag if needed
                    if len(token_buffer) > 0:
                        print(
                            f"[DEBUG] Immediate flush due to termination flag in buffer (size {len(token_buffer)})."
                        )
                        buffer_format = process_format_to_response(
                            token_buffer,
                            qry_id,
                            continue_="E",
                            update_index=answer_counter,
                        )
                        send_data_to_server(buffer_format, response_url)
                        token_buffer = []
                    break
            # After loop: if tokens remain, flush them with termination flag
            if len(token_buffer) > 0:
                print(
                    f"[DEBUG] Final flush of remaining {len(token_buffer)} tokens with end flag."
                )
                buffer_format = process_format_to_response(
                    token_buffer, qry_id, continue_="E", update_index=answer_counter
                )
                send_data_to_server(buffer_format, response_url)
        except Exception as e:
            print(f"[ERROR] sse_generator encountered an error: {e}")
        finally:
            try:
                ray.get(sse_manager.delete_queue.remote(request_id))
            except Exception as ex:
                print(f"[DEBUG] Error closing SSE queue for {request_id}: {str(ex)}")
            print("[DEBUG] SSE closed.")

    # 별도의 스레드에서 SSE generator 실행
    job = threading.Thread(
        target=sse_generator, args=(request_id, response_url), daemon=False
    )
    job.start()

    # 클라이언트에는 즉시 "수신양호" 메시지를 JSON 형식으로 응답
    return Response(
        error_format("수신양호", 200, qry_id), content_type="application/json"
    )


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
        response = inference_handle.get_history.remote(
            request_id, last_index=last_index
        )
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


# 새로 추가3: request_id로 해당 답변 생성을 중도에 멈출 수 있는 API
@app.route("/stop_generation", methods=["POST"])
def stop_generation():
    """
    Endpoint to stop an ongoing generation process
    """
    try:
        body = request.json or {}
        request_id = body.get("request_id")

        if not request_id:
            error_resp = error_format("request_id parameter is required", 400)
            return Response(error_resp, content_type="application/json; charset=utf-8")

        print(f"[DEBUG] Received stop generation request for request_id={request_id}")

        # Send the stop signal to the inference service
        response = inference_handle.stop_generation.remote(request_id)
        obj_ref = response._to_object_ref_sync()
        result = ray.get(obj_ref)

        # Return success response
        return jsonify(
            {
                "status": "success",
                "message": f"Stop request for {request_id} received",
                "detail": result,
            }
        )

    except Exception as e:
        error_resp = error_format(f"Error processing stop request: {str(e)}", 500)
        return Response(error_resp, content_type="application/json; charset=utf-8")


# --------------------- 프롬프트 테스트 엔드포인트 ----------------------------
@app.route("/test_prompt", methods=["POST"])
async def test_prompt():
    """
    새 프롬프트를 테스트하는 API 엔드포인트.

    필요한 파라미터:
    - prompt: 테스트할 프롬프트
    - user_input: 사용자 입력 텍스트
    - file_data: (선택) 파일 데이터 (base64 인코딩)
    - file_type: (선택) 파일 타입 ('image' 또는 'pdf')
    """
    try:
        # 요청 파라미터 추출
        body = request.json or {}
        system_prompt = body.get("prompt", "")
        user_input = body.get("user_input", "")
        file_data = body.get("file_data")
        file_type = body.get("file_type", "image")  # 기본값은 이미지

        if not system_prompt:
            return Response(
                error_format("프롬프트는 필수입니다.", 400), content_type=content_type
            )

        print(
            f"[DEBUG] /test_prompt called with prompt length={len(system_prompt)}, user_input='{user_input[:50]}...' and file_type={file_type}"
        )

        # 요청 ID 생성
        request_id = str(uuid.uuid4())

        # Ray Serve를 통한 프롬프트 테스트 호출
        response = inference_handle.test_prompt.remote(
            system_prompt, user_input, file_data, file_type, request_id
        )
        obj_ref = response._to_object_ref_sync()
        result = ray.get(obj_ref)

        # 결과 반환
        return Response(
            json.dumps({"status": "success", "result": result}, ensure_ascii=False),
            content_type=content_type,
        )

    except Exception as e:
        error_resp = error_format(f"프롬프트 테스트 중 오류: {str(e)}", 500)
        return Response(error_resp, content_type=content_type)


# --------------------- 프롬프트 스트리밍 테스트 엔드포인트 ----------------------------
@app.route("/test_prompt_stream", methods=["POST"])
def test_prompt_stream():
    """
    새 프롬프트를 스트리밍 방식으로 테스트하는 API 엔드포인트.

    필요한 파라미터:
    - prompt: 테스트할 프롬프트
    - user_input: 사용자 입력 텍스트
    - file_data: (선택) 파일 데이터 (base64 인코딩)
    - file_type: (선택) 파일 타입 ('image' 또는 'pdf')
    - use_rag: (선택) RAG 사용 여부 (기본값: true)
    """
    try:
        # 요청 파라미터 추출
        body = request.json or {}
        system_prompt = body.get("prompt", "")
        user_input = body.get("user_input", "")
        file_data = body.get("file_data")
        file_type = body.get("file_type", "image")  # 기본값은 이미지
        use_rag = body.get("use_rag", True)  # RAG 사용 여부 (기본값: true)

        if not system_prompt:
            return Response(
                error_format("프롬프트는 필수입니다.", 400), content_type=content_type
            )

        print(
            f"[DEBUG] /test_prompt_stream called with prompt length={len(system_prompt)}, user_input='{user_input[:50]}...', file_type={file_type}, use_rag={use_rag}"
        )

        # 요청 ID 생성 (SSE 큐 ID로 사용)
        request_id = str(uuid.uuid4())

        # SSE 큐 생성
        ray.get(sse_manager.create_queue.remote(request_id))

        # Ray Serve를 통한 스트리밍 프롬프트 테스트 호출
        response = inference_handle.test_prompt_stream.remote(
            system_prompt, user_input, file_data, file_type, request_id, use_rag
        )
        obj_ref = response._to_object_ref_sync()
        chat_id = ray.get(obj_ref)

        def sse_generator():
            try:
                while True:
                    # SSEQueueManager에서 토큰을 가져옴
                    token = ray.get(sse_manager.get_token.remote(chat_id, 120))
                    if token is None or token == "[[STREAM_DONE]]":
                        break
                    yield f"data: {token}\n\n"
            except Exception as e:
                error_token = json.dumps({"type": "error", "message": str(e)})
                yield f"data: {error_token}\n\n"
            finally:
                try:
                    ray.get(sse_manager.delete_queue.remote(chat_id))
                except Exception as ex:
                    print(f"[DEBUG] Error closing SSE queue for {chat_id}: {str(ex)}")
                print("[DEBUG] SSE closed.")

        return Response(sse_generator(), mimetype="text/event-stream")

    except Exception as e:
        error_resp = error_format(f"프롬프트 스트리밍 테스트 중 오류: {str(e)}", 500)
        return Response(error_resp, content_type=content_type)

#  Flask 라우트들 아래에 추가 (stream 부분 이후 아무 곳이나 OK)
@app.route("/metrics", methods=["GET"])
async def get_metrics():
    try:
        snapshot = await inference_handle.metrics.remote()
        return Response(json.dumps(snapshot, ensure_ascii=False),
                        content_type="application/json; charset=utf-8")
    except Exception as e:
        return Response(json.dumps({"error": str(e)}, ensure_ascii=False),
                        status=500,
                        content_type="application/json; charset=utf-8")

@app.route("/global_metrics")
def global_metrics():
    try:
        coll = ray.get_actor("MetricsCollector")
        data = ray.get(coll.dump_global.remote())
        return Response(json.dumps(data, ensure_ascii=False),
                        content_type="application/json; charset=utf-8")
    except Exception as e:
        return Response(json.dumps({"error": str(e)}, ensure_ascii=False),
                        status=500, content_type="application/json; charset=utf-8")


# Flask app 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
