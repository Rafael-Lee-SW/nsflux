# sql.py
import json
import sqlite3
import re

def generate_sql(query, model, tokenizer, config):
    with open(config.metadata_path, 'r', encoding='utf-8') as file:
        Metadata = json.load(file)
    column_usage = Metadata['column_usage']

    # first_LLM
    outputs_1, filter_conditions, aggregations, orders, sql_query, parsed_columns = first_llm(model, tokenizer, column_usage, query, config)
    print(f'FirstLLM\n필터:{filter_conditions}\n집계:{aggregations}\n정렬:{orders}\nSQL:{sql_query}\n컬럼:{parsed_columns}')
    print(config.beep)

    relevant_metadata = extract_relevant_metadata(parsed_columns, column_usage) # 추출된 컬럼에 해당하는 메타데이터 가져오기
    retrival_metadata = parse_and_augment_filter_conditions(filter_conditions, Metadata)   # Metadata와 매핑하여 구체화된 필터 조건 찾기
    print(f'MetaData\n관련:{relevant_metadata}\n검색:{retrival_metadata}')
    print(config.beep)
    # second_LLM
    final_sql_query, title, explain, outputs_2 = second_llm(model, tokenizer, relevant_metadata, sql_query, query, retrival_metadata, parsed_columns, config)
    print(f'SecondLLM\n제목:{title}\n설명:{explain}\nSQL:{final_sql_query}')
    print(config.beep)
    
        # SQL 실행
    columns, results = execute_sql_query(final_sql_query, config)
    if columns is None and results is None:
        # DB 에러 발생 등
        print("[SQL Debug] DB 처리 중 오류가 발생했거나 SQL이 잘못되어 결과 없음.")
        return None

    # 결과 처리
    if not results:
        print("[SQL Debug] SQL 실행 결과가 비어 있습니다. (0개 행)")
        # None을 반환하여 상위에서 체크하게끔.
        return None

    # result -> json
    table_json = create_table_json(columns, results)
    chart_json = create_chart_json(columns, results)
    
    print("[SQL Debug] 조회된 컬럼:", columns)
    print("[SQL Debug] 조회된 결과(Row) 예시:", results[:3], "...")
    return final_sql_query, title, explain, table_json, chart_json

    
def first_llm(model, tokenizer, column_usage, user_query, config):
    PROMPT =\
    f'''
    <bos><start_of_turn>user
    너는 남성 해운 회사의 데이터로 SQL 쿼리를 작성하는 데 도움을 주는 시스템이야. 사용자로부터 받은 질문을 분석하여, 필터 조건, 집계 함수, 정렬 조건, SQL 쿼리 초안, SQL 쿼리에 사용된 모든 컬럼을 추출해줘.
    
    ### 참고 사항:
    1. 다음은 해운 회사 데이터의 메타데이터야. 테이블은 "revenue" 하나뿐이야.: 
    "{column_usage}"
    2. 사용자가 입력한 질문을 분석하여 필요한 컬럼을 식별하고, SQL 쿼리에서 사용할 필터 조건, 집계 함수, 정렬 기준을 제공해줘.
    3. 사용되는 프로그램은 SQLite 야. 이 프로그램에 맞는 언어를 사용해줘 (SQLite 날짜 형식 사용 예시 : strftime('%Y', OUTOBD) AS Year )
    
    ### 사용자가 입력한 질문:
    "{user_query}"
    
    ### 필요한 정보:
    1. 필터 조건 (필요한 경우, 예: <filter/>OUTPOL = '부산', OUTPOD = '일본', OUTBOR = '2024-08-01 이후'<filter/>)
    2. 집계 함수 (필요한 경우, 예: <aggregation/>화주(고객)별 매출액의 합계<aggregation/>)
    3. 정렬 조건 (필요한 경우, 예: <order/>매출액 기준 내림차순<order/>)
    4. SQL 쿼리 초안 (예: <sql_query/>SELECT OUTSHC,SUM(OUTSTL) AS TotalRevenue\n    FROM revenue\n    WHERE OUTPOL = \'한국\' AND OUTPOD = \'베트남\' AND OUTOBD >= \'2023-01-01\'    GROUP BY OUTSHC\n    ORDER BY TotalRevenue DESC;<sql_query/>)
    5. SQL 쿼리에 사용된 모든 컬럼 (예: <columns/>OUTPOL,OUTPOD,OUTBOR,OUTSHC,OUTSTL<columns/>)
    
    ### 출력 형식:
    1. 필터 조건: <filter/><filter/>
    2. 집계 함수:<aggregation/><aggregation/>
    3. 정렬 조건:<order/><order/>
    4. SQL 쿼리 초안 : <sql_query/><sql_query/>
    5. SQL 쿼리에 사용된 모든 컬럼:<columns/><columns/>
    6. 날짜는 YYYY-MM-DD 형식을 사용 (ex: "2023-05-01")
    
    <end_of_turn>
    <start_of_turn>model
    '''

    # Get Answer
    input_ids = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    input_length = input_ids['input_ids'].shape[1]
    outputs = model.generate(**input_ids, max_new_tokens=config.model.max_new_tokens)
    outputs_result = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    # Regular expression to extract content between <query/> and <query>
    filter_pattern = r'<filter.*?>(.*?)<filter.*?>'
    aggregation_pattern = r'<aggregation.*?>(.*?)<aggregation.*?>'
    order_pattern = r'<order.*?>(.*?)<order.*?>'
    sql_pattern = r'<sql_query.*?>(.*?)<sql_query.*?>'
    columns_pattern = r'<columns.*?>(.*?)<columns.*?>'
    
    filter_conditions = re.search(filter_pattern, outputs_result, re.DOTALL).group(1)
    aggregations = re.search(aggregation_pattern, outputs_result, re.DOTALL).group(1)
    orders = re.search(order_pattern, outputs_result, re.DOTALL).group(1)
    sql_queries = re.search(sql_pattern, outputs_result, re.DOTALL).group(1)
    parsed_columns = [col.strip() for col in re.search(columns_pattern, outputs_result, re.DOTALL).group(1).split(",")]
    return outputs_result, filter_conditions, aggregations, orders, sql_queries, parsed_columns

# 추출된 컬럼에 해당하는 메타데이터만 가져오는 함수
def extract_relevant_metadata(columns, metadata):
    relevant_metadata = {}
    for column in columns:
        if column in metadata["column_usage"]:
            relevant_metadata[column] = metadata["column_usage"][column]
    return relevant_metadata

# def parse_and_augment_filter_conditions(filter_conditions, Metadata):
#     pattern = r"(\w+)\s*=\s*'([^']+)'"  # ex: OUTPOL = '부산' 과 같은 패턴 추출
#     matches = re.findall(pattern, filter_conditions)
    
#     augmented_filters = []
    
#     for col, val in matches:
#         if col == 'OUTPOL' or col == 'OUTPOD':
#             location_code = Metadata['location_code']
#             mapped_value = search_location_db(val, location_code)
#             if mapped_value != "UNKNOWN":
#                 augmented_filters.append(f"컬럼 {col}에 대한 값 '{val}' -> '{mapped_value}'로 매핑되었습니다.")
#             else:
#                 augmented_filters.append(f"컬럼 {col}의 값 '{val}'에 대한 매핑 정보를 찾을 수 없습니다.")
#     return "\n".join(augmented_filters)

def parse_and_augment_filter_conditions(filter_conditions, Metadata):
    # '컬럼명 = '값'' 또는 '컬럼명 IN ('값1', '값2', ...)' 패턴에 대응하는 정규식
    pattern = r"(\w+)\s*=\s*'([^']+)'|\b(\w+)\s+IN\s+\(([^)]+)\)"
    matches = re.findall(pattern, filter_conditions)
    
    augmented_filters = []
    for match in matches:
        # 매칭 결과에서 'IN' 조건과 '=' 조건을 구분하여 처리
        if match[0]:  # '=' 조건
            col, val = match[0], match[1]
            if col == 'OUTPOL' or col == 'OUTPOD':
                location_code = Metadata['location_code']
                mapped_value = search_location_db(val, location_code)
                if mapped_value != "UNKNOWN":
                    augmented_filters.append(f"컬럼 {col}에 대한 값 '{val}' -> '{mapped_value}'로 매핑되었습니다.")
                else:
                    augmented_filters.append(f"컬럼 {col}의 값 '{val}'에 대한 매핑 정보를 찾을 수 없습니다.")
                    
        elif match[2]:  # 'IN' 조건
            col, val_list = match[2], match[3]
            if col == 'OUTPOL' or col == 'OUTPOD':
                location_code = Metadata['location_code']
                values = [val.strip().strip("'") for val in val_list.split(",")]
                
                mapped_values = []
                for val in values:
                    mapped_value = search_location_db(val, location_code)
                    if mapped_value != "UNKNOWN":
                        mapped_values.append(f"'{val}' -> '{mapped_value}'")
                    else:
                        mapped_values.append(f"'{val}' (매핑 정보 없음)")
                
                augmented_filters.append(f"컬럼 {col}에 대한 값들: {', '.join(mapped_values)}")

    return "\n".join(augmented_filters)

# Location 검색 알고리즘 (매핑 정보 검색)
def search_location_db(location, location_code):
    return location_code.get(location, "Mapping error")

def second_llm(model, tokenizer, relevant_metadata, sql_query, user_query, retrival_metadata, parsed_columns, config):
    PROMPT =\
    f'''
    <bos><start_of_turn>user
    너는 남성 해운 회사의 데이터로 정확한 SQL 쿼리를 작성해주는 시스템이야. 너가 참고해야 할 정보가 있는 경우에는 이를 참고해서 SQL 쿼리 초안을 구체화해서 정확한 SQL 쿼리를 만들어줘. 그리고 이 SQL 쿼리가 어떤 정보를 추출해주는지 짧게 제목을 짓고, 어떻게 사용자의 질문에 답할 수 있는 정보를 추출하는지 설명해줘. 참고해야 할 정보가 없고 SQL 쿼리 초안이 이미 정확하다면, 그대로 출력해줘.

    ### 참고 사항:
    1. 다음은 너가 참고해야 할 정보야:
    "{retrival_metadata}"
    2. SQL 쿼리 초안:
    "{sql_query}"    
    3. 다음은 사용한 데이터의 메타데이터야:
    "{relevant_metadata}"
    4. 다음은 사용자가 입력한 질문이야:
    "{user_query}"    


    ### 필요한 정보:
    1. 정확한 SQL 쿼리 (예: <sql_query/>SELECT OUTSHC,SUM(OUTSTL) AS TotalRevenue\n    FROM revenue\n    WHERE WHERE OUTPOL = \'KRPUS\' AND OUTPOD LIKE \'CN%\' AND OUTOBD >= \'2023-01-01\'\n    GROUP BY OUTSHC\n    ORDER BY TotalRevenue DESC;<sql_query/>)
    2. SQL가 조회하는 데이터 요약 (예: 부산발 중국착 매출 순위 (화주별))
    3. SQL 쿼리 설명

    ### 출력 형식(아래 출력 형식을 꼭 지켜야 해 시작부분에 / 이 들어가고 끝부분에는 없어):
    1. 정확한 SQL 쿼리: <sql_query/>SQL 명령어<sql_query>
    2. SQL가 조회하는 데이터 요약: <title/>데이터 설명문<title>
    3. SQL 쿼리 설명: <explain/>SQL 설명문<explain>
    
    ### 참고자료
    1. 만약 참고자료에 KR% 같은 조건이 있으면 LIKE 를, KRCRD 같은 정확한 정보는 = 를 사용.
    2. 만약 여러개의 LIKE 조건이 있으면 (예시: WHERE OUTPOL LIKE 'KR%' OR OUTPOL LIKE 'CN%' OR OUTPOL LIKE 'JP%') 를 사용.
    3. 날짜는 YYYY-MM-DD 형식을 사용 (ex: "2023-05-01")
    4. LIKE 로 들어간 컬럼들은 다음과 같이 보기좋게 해줘.
    예시 : 
    SELECT 
        CASE 
            WHEN OUTPOL LIKE 'KR%' THEN '한국'
            WHEN OUTPOL LIKE 'JP%' THEN '일본'
            WHEN OUTPOL LIKE 'CN%' THEN '중국'
            ELSE '기타' 
        END AS 국가,
        SUM(OUTSTL) AS TotalRevenue

    <end_of_turn>
    <start_of_turn>model
    '''

    # Get Answer
    input_ids = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    input_length = input_ids['input_ids'].shape[1]
    outputs = model.generate(**input_ids, max_new_tokens=config.model.max_new_tokens)
    outputs_result = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    print(f'2번째 LLM Output:{outputs_result}')
    sql_pattern = r'<sql_query.*?>(.*?)<sql_query.*?>'
    title_pattern = r'<title.*?>(.*?)<title.*?>'
    explain_pattern = r'<explain.*?>(.*?)<explain.*?>'
    
    sql_queries = re.search(sql_pattern, outputs_result, re.DOTALL).group(1)
    title = re.search(title_pattern, outputs_result, re.DOTALL).group(1)
    explain = re.search(explain_pattern, outputs_result, re.DOTALL).group(1)
    
    return sql_queries, title, explain, outputs_result

def execute_sql_query(sql_query, config):
    try:
        conn = sqlite3.connect(config.sql_data_path)        # SQLite 데이터베이스에 연결
        cursor = conn.cursor()

        if (config.k is not None) and ("LIMIT" not in sql_query):
            sql_query = sql_query.split(";")[0].strip()
            sql_query += f"\nLIMIT {config.k};"
                 
        cursor.execute(sql_query)        # SQL 쿼리 실행
        
        result = cursor.fetchall()        # 결과 가져오기
        column_names = [description[0] for description in cursor.description]        # 컬럼 이름도 포함하기 위해 description을 사용

        cursor.close()
        conn.close()
        
        return column_names, result

    except sqlite3.Error as e:
        print(f"데이터베이스 오류 발생: {e}")
        return None, None

def create_table_json(columns, results):
    head = "||".join(columns)
    body = "^ ".join("||".join(map(str, row)) for row in results)
    table = {"head": head, "body": body}
    return table

def create_chart_json(columns, results):
    chart = [
        {"label": f"{row[-2]}", "data": [{"x": "매출액", "y": str(row[-1])}]}
        for i, row in enumerate(results)
    ]
    return chart