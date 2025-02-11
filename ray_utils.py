# ray_utils.py

import ray
import json
import yaml

# Importing the Legacy code
from RAG import query_sort, execute_rag, generate_answer
from utils import load_data, process_format_to_response, process_to_format, error_format

# Ray Initialization Code (Already Exited init_ray() function from ray_setup still be maintain)
def init_ray():
    ray.init(
        include_dashboard=True,
        dashboard_host="0.0.0.0"  # Exteranl IP can access the server
    )
    print("Ray initialized. Dashboard running at http://<server-ip>:8265")

@ray.remote(num_gpus=1)
class InferenceActor:
    def __init__(self, kwargs):

        # 액터 초기화 시, 모델, 토크나이저 등 GPU 객체를 한 번 로드하고,
        # 데이터를 미리 로딩합니다.

        self.kwargs = kwargs
        # 초기 데이터 로드 (필요시 이후에도 새로 로드할 수 있음)
        self.kwargs["data"] = load_data(self.kwargs["config"].data_path)
    
    def process_query(self, http_query):
        try:
            # 클라이언트 요청에서 사용자 입력(query) 추출
            user_input = http_query.get('qry_contents', '')
            
            # 필요한 경우, 최신 데이터를 다시 로드 (예: 데이터가 자주 업데이트되는 경우)
            self.kwargs["data"] = load_data(self.kwargs["config"].data_path)
            
            # legacy RAG 시스템 호출: query_sort를 이용하여 구체화 질문, 키워드, 테이블 여부, 시간 범위를 구분
            QU, KE, TA, TI = query_sort(user_input, **self.kwargs)
            
            # 테이블(SQL) 여부에 따른 분기 처리
            if TA == "yes":
                try:
                    docs, docs_list = execute_rag(QU, KE, TA, TI, **self.kwargs)
                    retrieval, chart = process_to_format(docs_list, type="SQL")
                    output = generate_answer(QU, docs, **self.kwargs)
                    answer = process_to_format([output, chart], type="Answer")
                    outputs = process_format_to_response(retrieval, answer)
                except Exception as e:
                    return error_format("내부 Excel 에 해당 자료가 없습니다.", 551)
            else:
                try:
                    docs, docs_list = execute_rag(QU, KE, TA, TI, **self.kwargs)
                    retrieval = process_to_format(docs_list, type="Retrieval")
                    output = generate_answer(QU, docs, **self.kwargs)
                    answer = process_to_format([output], type="Answer")
                    outputs = process_format_to_response(retrieval, answer)
                except Exception as e:
                    return error_format("내부 PPT에 해당 자료가 없습니다.", 552)
            
            response = json.dumps(outputs, ensure_ascii=False)
            return response
        except Exception as e:
            return error_format(f"처리 중 오류 발생: {str(e)}", 500)