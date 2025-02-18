from flask import Flask, render_template, request, Response
from RAG import generate_answer, execute_rag, query_sort  # 기존에 만든 RAG 시스템 불러오기
import json
import yaml
from box import Box
from utils import load_model, load_data, random_seed, process_format_to_response, process_to_format, error_format
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import time
from datetime import datetime, timedelta

# Configuration
with open('./config.yaml', 'r') as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)
random_seed(config.seed)

# Load Model
model, tokenizer, embed_model, embed_tokenizer = load_model(config)

def answer_format(status_code, continue_, qry_id, ans):
    ans_format = {
            "status_code": status_code,
            "result": "OK",
            "detail": "",
            "continue": continue_,
            "qry_id": qry_id,
            "rsp_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "data_list": ans,
        }
    return ans_format

def run_generation(user_text):
    # 모델 입력 준비
    model_inputs = tokenizer([user_text], return_tensors="pt").to('cuda')
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=5000)

    # 모델에서 텍스트 생성을 백그라운드 스레드로 실행
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # SSE 형식으로 각 새 텍스트 토큰을 클라이언트에 전송
    model_output = ""
    # for new_text in streamer:
    #     model_output += new_text
    #     new_text_form = process_to_format([new_text], type="Answer")
    #     response = answer_format(200, "C", "20240000", new_text_form)
    #     yield json.dumps(response).encode('utf-8')  # SSE 형식으로 스트리밍
    #     print(new_text)
    #     time.sleep(0.001)
        
    # model_output_form = process_to_format([model_output], type="Answer")
    # output_form = answer_format(200, "E", "20240000", model_output_form)
    # yield json.dumps(output_form).encode('utf-8')
    for i in range(10):
        yield i.encode("utf-8")
        print(i)
        time.sleep(0.1)
########## FLASK APP ##########
app = Flask(__name__)

# 기본 페이지를 불러오는 라우트
@app.route('/')
def index():
    return render_template('index_socket.html')

# WebSocket 이벤트 처리
@app.route('/generate', methods=['GET'])
def generate():
    user_text = request.args.get('input_text', '')
    return Response(run_generation(user_text), mimetype='text/event-stream')

@app.route('/queryToSLLM', methods=['POST'])
def query():
    http_query = request.json  # JSON 형식으로 query를 받음
    user_input = http_query.get('qry_contents', '')
    print(f"질문 : {user_input}")

    # 여기서 query_id를 생성 (예: UUID나 특정 형식으로 생성)
    query_id = "20240000"  # 예시로 고정된 query_id 사용, 필요에 따라 변경
    
    # 클라이언트에게 query_id 반환
    return json.dumps({"qry_id": query_id})

@app.route('/queryToSLLM/stream/<query_id>', methods=['GET'])
def stream_response(query_id):
    # query_id를 이용해 적절한 처리를 해야 할 경우 추가 로직을 넣을 수 있습니다.
    return Response(run_generation(query_id), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
