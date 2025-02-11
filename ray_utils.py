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
def process_query_remote(http_query, kwargs_serialized):
    # Remote Ray function. Does not touch the legacy code, just call the function
    try:
        # kwargs_serialized는 직렬화된 객체이므로, 원래 객체로 복원합니다.
        # (kwargs는 딕셔너리로 구성되어 있으며, 모델/토크나이저 등 GPU 객체가 포함되어 있음)
        # 이 예제에서는 단순 전달한다고 가정하고, 필요에 따라 pickle 또는 ray.put으로 공유할 수 있습니다.
        kwargs = kwargs_serialized
        
        # 요청 JSON에서 사용자 입력(query) 추출
        user_input = http_query.get('qry_contents', '')
        
        # 데이터 재로딩 (원래 app.py에서도 로딩하는 방식과 동일)
        data = load_data(kwargs["config"].data_path)
        kwargs["data"] = data
        
        # legacy RAG 시스템을 이용한 질의 분류 (구체화 질문, 키워드, 테이블 여부, 시간 범위)
        QU, KE, TA, TI = query_sort(user_input, **kwargs)
        
        # 테이블(SQL) 여부에 따른 분기 처리
        if TA == "yes":
            try:
                docs, docs_list = execute_rag(QU, KE, TA, TI, **kwargs)
                retrieval, chart = process_to_format(docs_list, type="SQL")
                output = generate_answer(QU, docs, **kwargs)
                answer = process_to_format([output, chart], type="Answer")
                outputs = process_format_to_response(retrieval, answer)
            except Exception as e:
                return error_format("내부 Excel 에 해당 자료가 없습니다.", 551)
        else:
            try:
                docs, docs_list = execute_rag(QU, KE, TA, TI, **kwargs)
                retrieval = process_to_format(docs_list, type="Retrieval")
                output = generate_answer(QU, docs, **kwargs)
                answer = process_to_format([output], type="Answer")
                outputs = process_format_to_response(retrieval, answer)
            except Exception as e:
                return error_format("내부 PPT에 해당 자료가 없습니다.", 552)
        
        response = json.dumps(outputs, ensure_ascii=False)
        return response
    except Exception as e:
        return error_format(f"처리 중 오류 발생: {str(e)}", 500)
