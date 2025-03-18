# SQL_NS.py

import os
import subprocess
from utils.tracking import time_tracker
import json
from utils.utils import load_model
import re

# 환경 변수 설정
os.environ['ORACLE_HOME'] = '/workspace/oracle/instantclient_23_7'
os.environ['LD_LIBRARY_PATH'] = os.environ['ORACLE_HOME'] + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PATH'] = os.environ['ORACLE_HOME'] + ':' + os.environ.get('PATH', '')

import yaml
from box import Box

# Config 불러오기
with open("./config.yaml", "r") as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)

# 기본 SQL 접속코드
sqlplus_command = [
            'sqlplus', '-S', 'LLM/L9SD2TT9XJ0H@//210.113.16.230:1521/ORA11GDR'
        ]

'''
### ORACLE DB 정보 ###
TABLE : ai_dg_check
    COLUMNS : CLS (위험물 클래스)
              UNNO (위험물 UN 번호)
              PORT (포트 번호)
              ALLOW_YN (취급 가능 여부)
'''

SQL_UNNO_PROMPT = \
"""
<bos>
<system>
너는 남성해운의 내부 데이터를 기반으로 질문에 답하는 데이터 분석가야.
- 문서를 바탕으로 사실적인 답변을 한다.
- 문서에 없는 내용은 "내부 자료에 해당 자료 없음"이라고 명시한다.
- 표 데이터를 말로 풀어 해석한 뒤 인사이트를 제공한다.
- 출처 표기는 필수다.
</system>

<user>
내부 자료: {docs}
질문: {query}
</user>

<assistant>
답변:
</assistant>
"""

# 테스트를 위한 초기설정
def initialze(config):
    

    model, tokenizer, _, _ = load_model(config)
    return model, tokenizer

# sqlplus 실행 여부 확인
def check_sqlplus():
    try:
        # sqlplus 버전 확인
        result = subprocess.run(['sqlplus', '-version'], capture_output=True, text=True, check=True)
        print(" SQL*Plus is working!")
        print("Version info:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")

# DB 연결상태 확인
def check_db_connection():
    try:
        # SQL*Plus 실행 명령
        
        # SQL 명령을 표준 입력으로 전달
        sql_query = "SELECT 1 FROM dual;\nEXIT;\n"
        result = subprocess.run(
            sqlplus_command,
            input=sql_query,  # SQL 명령을 표준 입력으로 전달
            capture_output=True,
            text=True
        )
        
        # SQL*Plus 결과 분석
        if "1" in result.stdout:
            print("  Successfully connected to the Namsung database!")
        else:
            print(" Connection to the database failed!")

    except subprocess.CalledProcessError as e:
        print(f" Error: {e.stderr}")

# 스키마 별 테이블 목록 출력
@time_tracker
def get_all_schema_tables():
    try:
        # SQL*Plus 실행 명령
        sqlplus_command = [
            'sqlplus', '-S', 'LLM/L9SD2TT9XJ0H@//210.113.16.230:1521/ORA11GDR'
        ]

        # SQL 실행 (스키마별 테이블 목록 조회)
        sql_query = """SET PAGESIZE 0 FEEDBACK OFF VERIFY OFF HEADING OFF ECHO OFF;
        SELECT OWNER, TABLE_NAME FROM ALL_TABLES ORDER BY OWNER, TABLE_NAME;
        EXIT;"""

        # SQL*Plus 실행
        result = subprocess.run(
            sqlplus_command,
            input=sql_query,
            capture_output=True,
            text=True
        )

        # 결과 분석
        schema_tables = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                parts = line.split()  # 공백 기준으로 OWNER와 TABLE_NAME 분리
                if len(parts) >= 2:
                    schema, table = parts[0], parts[1]
                    if schema not in schema_tables:
                        schema_tables[schema] = []
                    schema_tables[schema].append(table)

        # 결과 출력
        if schema_tables:
            print("  스키마별 테이블 목록:")
            for schema, tables in schema_tables.items():
                print(f"\n🔹 스키마: {schema}")
                for table in tables:
                    print(f"  - {table}")
        else:
            print(" 테이블이 존재하지 않습니다.")

        return schema_tables

    except subprocess.CalledProcessError as e:
        print(f" Error: {e.stderr}")
        return {}

# OPRAIMDG에서 메타데이터 만들기.
def make_metadata_from_table(schema_name="ICON", table_name="OPRAIMDG"):
    
    # LINESIZE : 컬럼이 길때 다음줄로 출력하는 것 방지
    # PAGESIZE : 0일 경우 헤더 무시
    # TRIMSPOOL : 의미없는 공백 무시
    # IMDCOM FORMAT A200 : IMDCOM 컬럼의 출력 길이 늘리기
    sql_query = f"""
    SET LINESIZE 2000;
    SET PAGESIZE 0;
    SET TRIMSPOOL ON;
    COL IMDCOM FORMAT A200;
    -- 개행문자 없애기
    SELECT IMDUNM, IMDCLS, REPLACE(REPLACE(IMDCOM, CHR(10), ' '), CHR(13), ' ') AS IMDCOM 
    FROM {schema_name}.{table_name};
    EXIT;
    """
    
    try:
        # SQL*Plus 실행 및 결과 캡처
        result = subprocess.run(sqlplus_command, input=sql_query, capture_output=True, text=True)
        print(f"  RESULT: \n{str(result)[:1000]}")
        output = result.stdout
        print(f"  OUTPUT: \n{str(output)[:1000]}")
        
        # 결과 파싱
        lines = output.strip().split("\n")
        print(f"  LINE: \n{str(lines)[:1000]}")
        metadata = []
        
        for line in lines[:-1]:
            # print(line)
            values = line.split(None, 2)  # 첫 두 개는 그대로, 세 번째는 나머지 전체를 포함
            if len(values) == 3:
                imdunm = values[0].strip()
                imdcls = values[1].strip()
                imdcom = values[2].strip()  # 설명은 전체 유지
                metadata.append({
                    "UNNO": imdunm,
                    "Class": imdcls,
                    "Description": imdcom
                })
        
        # JSON 파일로 저장
        json_filename = "/workspace/data/METADATA_OPRAIMDG.json"
        with open(json_filename, "w", encoding="utf-8") as json_file:
            json.dump(metadata, json_file, indent=4, ensure_ascii=False)
        
        print(f"  Metadata saved to {json_filename}")
    
    except subprocess.CalledProcessError as e:
        print(f" SQL Execution Error: {e.stderr}")

# # Oracle sqlplus 명령어 실행 예시
@time_tracker
def run_sql_unno(cls=None, unno=None, pol_port='KR%', pod_port='JP%'):
    # 값이 "NULL"이 아니면 문자열로 취급하여 작은따옴표로 감쌈.
    cls_val = "NULL" if (cls is None or cls == "NULL") else f"'{cls}'"
    unno_val = "NULL" if (unno is None or unno == "NULL") else f"'{unno}'"

    # SQL*Plus 명령어를 실행할 기본 명령어
    sql_query = \
    f"""
    SET LINESIZE 150;
    SET PAGESIZE 1000;
    SET TRIMSPOOL ON;

    SELECT 
        p.cls  AS CLS,
        p.unno AS UNNO,
        p.port AS POL_PORT,
        d.port AS POD_PORT,
        DECODE(p.allow_yn,'Y','OK','N','Forbidden','Need to contact PIC of POL') AS Landing_STATUS,
        DECODE(d.allow_yn,'Y','OK','N','Forbidden','Need to contact PIC of POL') AS Departure_STATUS
    FROM icon.ai_dg_check p
    JOIN icon.ai_dg_check d 
        ON p.unno = d.unno 
        AND p.cls = d.cls
    WHERE (p.cls={cls_val} OR {cls_val} IS NULL) AND (p.unno={unno_val} OR {unno_val} IS NULL) AND p.port LIKE '{pol_port}'
      AND (p.cls={cls_val} OR {cls_val} IS NULL) AND (d.unno={unno_val} OR {unno_val} IS NULL) AND d.port LIKE '{pod_port}';
    EXIT;
    """
    
    # subprocess를 사용하여 SQL*Plus 명령어 실행
    try:
        result = subprocess.run(sqlplus_command, input=sql_query, capture_output=True, text=True)
        # SQL*Plus의 출력 결과를 받아옵니다
        print("  SQL Query Results:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        # 오류가 발생한 경우 오류 메시지 출력
        print(f" Error: {e.stderr}")
    # import code
    # code.interact(local=locals())  # 현재 변수들을 유지한 상태에서 Python 인터랙티브 셸 실행
    return sql_query, result.stdout

def get_metadata(config):
    """
    - port_path JSON: 딕셔너리 형태이며, 'location_code' 키의 값을 추출.
    - unno_path JSON: 리스트 형태이며, 모든 항목을 문자열로 반환.
    """
    print("[SOOWAN] get_metadata 진입")
    print("[SOOWAN] get_metadata 진입")
    if not config or not hasattr(config, "metadata_unno"):
        raise ValueError("Config 객체에 'metadata_unno' 속성이 없습니다. config: {}".format(config))
    unno_path = config.metadata_unno
    port_path = config.metadata_path

    # port_path JSON 파일 로드 (딕셔너리)
    with open(port_path, "r", encoding="utf-8") as f:
        port_data = json.load(f)
    
    # location_code 값 추출 (키가 없을 경우 빈 리스트 반환)
    location_codes = json.dumps(port_data.get("location_code"), ensure_ascii=False)

    # unno_path JSON 파일 로드 (리스트)
    with open(unno_path, "r", encoding="utf-8") as f:
        unno_data = json.load(f)
    
    # 리스트 내 모든 요소를 문자열로 변환
    unno_list_as_string = json.dumps(unno_data, ensure_ascii=False)

    return location_codes, unno_list_as_string


@time_tracker
async def generate_sql(user_query, model, tokenizer, config):
    
    # Parse Metadata
    metadata_location, metadata_unno = get_metadata(config)
    # metadata_location = get_metadata(config)

    PROMPT =\
f'''
<bos>
<system>
"YourRole": "질문으로 부터 조건을 추출하는 역할",
"YourJob": "아래 요구 사항에 맞추어 'unno', 'class', 'pol_port', 'pod_port' 정보를 추출하여, 예시처럼 답변을 구성해야 합니다.",
"Requirements": [
    unno: UNNO Number는 4개의 숫자로 이루어진 위험물 번호 코드야. 
    class : UN Class는 2.1, 6.0,,, 의 숫자로 이루어진 코드야.
    pol_port, pod_port: 항구 코드는 5개의 알파벳 또는 나라의 경우 2개의 알파벳과 %로 이루어져 있어. 다음은 항구 코드에 대한 메타데이터야 {metadata_location}. 여기에서 매칭되는 코드만을 사용해야 해. 항구는 항구코드, 나라는 2개의 나라코드와 %를 사용해.
    unknown : 질문에서 찾을 수 없는 정보는 NULL을 출력해줘.
]

"Examples": [
    "질문": "UN 번호 1689 화물의 부산에서 미즈시마로의 선적 가능 여부를 확인해 주세요.",
    "답변": "<unno/>1689<unno>\\n<class/>NULL<class>\\n<pol_port/>KRPUS<pol_port>\\n<pod_port/>JPMIZ<pod_port>"

    "질문": "UN 클래스 2.1 화물의 한국에서 일본으로의 선적 가능 여부를 확인해 주세요.",
    "답변": "<unno/>NULL<unno>\\n<class/>2.1<class>\\n<pol_port/>KR%<pol_port>\\n<pod_port/>JP%<pod_port>"
]
- 최종 출력은 반드시 다음 4가지 항목을 포함해야 합니다:
    <unno/>...<unno>
    <class/>...<class>
    <pol_port/>...<pol_port>
    <pod_port/>...<pod_port>
</system>

<user>
질문: "{user_query}"
</user>

<assistant>
답변:
</assistant>
'''

    # --- 토큰 수 계산 단계 추가 ---
    tokenized_prompt = tokenizer(PROMPT, return_tensors="pt", truncation=True)
    token_count = tokenized_prompt["input_ids"].shape[1]
    print(f"[DEBUG] 프롬프트 토큰 수: {token_count}")

    # Get Answer
    ## From Vllm Inference
    from vllm import SamplingParams
    import uuid
    from core.RAG import collect_vllm_text
    sampling_params = SamplingParams(
        max_tokens=config.model.max_new_tokens,
        temperature=config.model.temperature,
        top_k=config.model.top_k,
        top_p=config.model.top_p,
        repetition_penalty=config.model.repetition_penalty,
    )
    # 성공할 때까지 최대 3회 반복
    max_attempts = 3
    attempt = 0
    UN_number = UN_class = POL = POD = "NULL"
    unno_pattern = r'<unno.*?>(.*?)<unno.*?>'
    class_pattern = r'<class.*?>(.*?)<class.*?>'
    pol_port_pattern = r'<pol_port.*?>(.*?)<pol_port.*?>'
    pod_port_pattern = r'<pod_port.*?>(.*?)<pod_port.*?>'

    while attempt < max_attempts:
        accepted_request_id = str(uuid.uuid4())
        outputs_result = await collect_vllm_text(PROMPT, model, sampling_params, accepted_request_id)
        print(f"[GENERATE_SQL] Attempt {attempt+1}, SQL Model Outputs: {outputs_result}")

        match_unno = re.search(unno_pattern, outputs_result, re.DOTALL)
        UN_number = match_unno.group(1).strip() if match_unno is not None else "NULL"

        match_class = re.search(class_pattern, outputs_result, re.DOTALL)
        UN_class = match_class.group(1).strip() if match_class is not None else "NULL"

        match_pol = re.search(pol_port_pattern, outputs_result, re.DOTALL)
        POL = match_pol.group(1).strip() if match_pol is not None else "NULL"

        match_pod = re.search(pod_port_pattern, outputs_result, re.DOTALL)
        POD = match_pod.group(1).strip() if match_pod is not None else "NULL"

        print(f"[GENERATE_SQL] 추출 결과 - UN_number: {UN_number}, UN_class: {UN_class}, POL: {POL}, POD: {POD}")

        # 조건: UN_number와 UN_class 중 하나라도 NULL이 아니고, POL과 POD는 모두 NULL이 아니어야 함.
        if ((UN_number != "NULL" or UN_class != "NULL") and POL != "NULL" and POD != "NULL"):
            break
        attempt += 1

    print(f"[GENERATE_SQL] 최종 추출 값 - UN_number: {UN_number}, UN_class: {UN_class}, POL: {POL}, POD: {POD}")
    final_sql_query, result = run_sql_unno(UN_class, UN_number, POL, POD)
    # Temporary: title, explain, table_json, chart_json은 None으로 처리
    title, explain, table_json, chart_json = (None,) * 4
    return final_sql_query, title, explain, result, chart_json

if __name__ == "__main__":
    # check_sqlplus()             # sqlplus가 잘 동작하는지 확인
    # check_db_connection()       # 데이터베이스 접속 여부 확인
    # get_all_schema_tables()    # ICON Table Name 반환
    # run_sql_unno(cls=4.1, pol_port="KR%", pod_port="JPUKB")         # 실제 SQL 쿼리 실행
    # make_metadata_from_table()

    query = "UN번호 1033, UN 클래스 2.1인 화물의 부산항에서 고베항으로의 선적이 가능한지 알아봐줘."
    model,tokenizer = initialze(config)
    # print(f"  METADATA: {metadata_location}")
    final_sql_query, title, explain, table_json, chart_json = generate_sql(query, model, tokenizer, config)
    print(f"  Final Sql Query: {final_sql_query}\n  Result: {table_json}")