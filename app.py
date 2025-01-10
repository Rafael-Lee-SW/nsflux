from flask import Flask, request, jsonify, render_template, Response
from RAG import generate_answer, execute_rag, query_sort  # 기존에 만든 RAG 시스템 불러오기
import json
import yaml
from box import Box
from utils import load_model, load_data, random_seed, process_format_to_response, process_to_format, error_format
import threading

# Configuration
with open('./config.yaml', 'r') as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)
random_seed(config.seed)

# Load Model
model, tokenizer, embed_model, embed_tokenizer = load_model(config)

# Load Data
data = load_data(config.data_path)

kwargs = {
    "model": model,
    "tokenizer": tokenizer,
    "embed_model": embed_model,
    "embed_tokenizer": embed_tokenizer,
    "data": data,
    "config": config,
}

########## FLASK APP ##########
app = Flask(__name__)
lock = threading.Lock()  # 전역 잠금 객체 생성
content_type='application/json; charset=utf-8'

# 기본 페이지를 불러오는 라우트
@app.route('/')
def index():
    return render_template('index.html')  # index.html을 렌더링

@app.route('/query', methods=['POST'])
def query():
    # 잠금을 사용하여 동시에 하나의 요청만 처리
    if not lock.acquire(blocking=False):
        return Response(error_format( "현재 앱이 사용중입니다. 잠시 후에 시도해주세요.", 550),
                        content_type=content_type)

    lock_acquired = True

    try:
        http_query = request.json  # JSON 형식으로 query를 받음
        user_input = http_query.get('qry_contents', '')
        
        # 기존 RAG 시스템을 이용해 답변 생성
        QU,KE,TA,TI = query_sort(user_input, **kwargs) # 구체화 질문, 키워드, 테이블 유무, 시간 범위

        if TA == "yes": # SQL 실행
            try:
                docs, docs_list = execute_rag(QU,KE,TA,TI, **kwargs) # docs 는 LLM 의 다음 input, docs_list 는 보여줄 정보들
                retrieval, chart = process_to_format(docs_list, type="SQL")

                output = generate_answer(QU, docs, **kwargs)
                answer = process_to_format([output, chart], type="Answer")
                outputs = process_format_to_response(retrieval, answer)
            except Exception as e:
                return Response(error_format(f"내부 Excel 에 해당 자료가 없습니다.", 551),
                                content_type=content_type)
            # finally:
            #     lock.release()
            #     lock_acquired = False

        elif TA == "no": # RAG 실행

            # RAG 실행 시 데이터 다시로드
            # Load Data
            data = load_data(config.data_path)

            kwargs = {
                "model": model,
                "tokenizer": tokenizer,
                "embed_model": embed_model,
                "embed_tokenizer": embed_tokenizer,
                "data": data,
                "config": config,
            }

            try:
                docs, docs_list = execute_rag(QU,KE,TA,TI, **kwargs) # RAG  실행
                retrieval = process_to_format(docs_list, type="Retrieval")
                
                output = generate_answer(QU, docs, **kwargs)
                answer = process_to_format([output], type="Answer")
                outputs = process_format_to_response(retrieval, answer)
            except Exception as e:
                return Response(error_format(f"내부 PPT에 해당 자료가 없습니다.", 552),
                                content_type=content_type)
            # finally:
            #     lock.release()
            #     lock_acquired = False

        # 결과를 JSON 형식으로 반환
        response = json.dumps(outputs, ensure_ascii=False)
        response = Response(response, content_type=content_type)
        return response
    
    finally:
        if lock_acquired:
            lock.release()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug = False)  # 서버 실행