from flask import Flask, request, jsonify, render_template, Response
from RAG import generate_answer  # 기존에 만든 RAG 시스템 불러오기
import json
import yaml
from box import Box
from utils import load_model, load_data, random_seed
from datetime import datetime

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

def process_query(qry_contents):
    # 여기서 RAG 시스템을 호출하거나 답변을 생성하도록 구현하세요.
    # 예제 응답 형식

    ans_format = {
        "status_code": 200,
        "result": "OK",
        "detail": "",
        "evt_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "data_list": [
            {"rsp_type": "TT", "rsp_tit": "답변", "rsp_data": qry_contents},
            {"rsp_type": "TB", "rsp_tit": "표 요약 내용", 
             "rsp_data": {
                 "head": "item1,item2,item3,item4",
                 "body": "data1,data2,data3,data4^ data1,data2,data3,data4^data1,data2,data3,data4"
             }},
            {"rsp_type": "CT", "rsp_tit": "차트 요약 내용", 
             "rsp_data": {
                 "chart_tp": "LINE",
                 "chart_data": [
                     {"series": "s1", "data": [{"x": "1", "y": "2"}, {"x": "2", "y": "3"}]},
                     {"series": "s2", "data": [{"x": "1", "y": "2"}, {"x": "2", "y": "3"}]},
                     {"series": "s3", "data": [{"x": "1", "y": "2"}, {"x": "2", "y": "3"}]}
                 ]
             }}
        ]
    }

    tmp_format = {
        "status_code": 200,
        "result": "OK",
        "detail": "",
        "evt_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "data_list": [
            {"rsp_type": "TT", "rsp_tit": "답변", "rsp_data": qry_contents},
        ]
    }
    return tmp_format

# 기본 페이지를 불러오는 라우트
@app.route('/')
def index():
    return render_template('index.html')  # index.html을 렌더링

@app.route('/query', methods=['POST'])
def query():
    http_query = request.json  # JSON 형식으로 query를 받음
    user_input = http_query.get('input', '')
    
    # 기존 RAG 시스템을 이용해 답변 생성
    output = generate_answer(model, tokenizer, embed_model, embed_tokenizer, data, user_input, config)
    
    # 답변 포맷 후처리
    output = process_query(output)

    # 결과를 JSON 형식으로 반환
    response = json.dumps({'output': output}, ensure_ascii=False)
    output = Response(response, content_type='application/json; charset=utf-8')
    return output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug = False)  # 서버 실행