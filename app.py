from flask import Flask, request, jsonify, render_template, Response
from RAG import generate_answer  # 기존에 만든 RAG 시스템 불러오기
import json
import yaml
from box import Box
from utils import load_model, load_data, random_seed, process_format_to_response, process_to_format

# Configuration
with open('./config.yaml', 'r') as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)
random_seed(config.seed)

# Load Model
model, tokenizer, embed_model, embed_tokenizer = load_model(config)

# Load Data
data = load_data(config.data_path)



########## FLASK APP ##########
app = Flask(__name__)

# 기본 페이지를 불러오는 라우트
@app.route('/')
def index():
    return render_template('index.html')  # index.html을 렌더링

@app.route('/query', methods=['POST'])
def query():
    http_query = request.json  # JSON 형식으로 query를 받음
    user_input = http_query.get('qry_contents', '')
    
    # 기존 RAG 시스템을 이용해 답변 생성
    output, docs_list = generate_answer(model, tokenizer, embed_model, embed_tokenizer, data, user_input, config)
    
    # 답변 포맷 후처리
    docs = process_to_format(docs_list, type="R")
    output = process_to_format([output], type="A")
    outputs = process_format_to_response(docs,output)

    # 결과를 JSON 형식으로 반환
    response = json.dumps({'output': outputs}, ensure_ascii=False)
    response = Response(response, content_type='application/json; charset=utf-8')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug = False)  # 서버 실행