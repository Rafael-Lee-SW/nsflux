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

# Configuration
with open('./config.yaml', 'r') as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)
random_seed(config.seed)

# Load Model
model, tokenizer, embed_model, embed_tokenizer = load_model(config)

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
    for new_text in streamer:
        model_output += new_text
        yield f"data: {new_text}\n\n"  # SSE 형식으로 스트리밍
        time.sleep(0.01)

    return model_output

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6460, debug=False)
