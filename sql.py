import json
import sqlite3
import re

def generate_sql(query, model, tokenizer, config):
    with open(config.metadata_path, 'r', encoding='utf-8') as file:
        Metadata = json.load(file)
    column_usage = Metadata['column_usage']

    # first_LLM 출력
    outputs_1 = first_llm(model, tokenizer, column_usage, query)

    parsed_columns = extract_columns_from_llm_output(outputs_1)   # 사용할 컬럼 추출

    relevant_metadata = extract_relevant_metadata(parsed_columns, column_usage) # 추출된 컬럼에 해당하는 메타데이터 가져오기
    sql_query = extract_sql_query(outputs_1)   # SQL 구문 초안 추출
    filter_conditions = extract_filter_conditions(outputs_1)   # 필터 조건 추출
    retrival_metadata = parse_and_augment_filter_conditions(filter_conditions, Metadata)   # Metadata와 매핑하여 구체화된 필터 조건 찾기

    # second_LLM 출력
    outputs_2 = second_llm(model, tokenizer, relevant_metadata, sql_query, retrival_metadata)
    final_sql_query = extract_sql_query(outputs_2)   # 최종 SQL 쿼리 생성
    print(final_sql_query)
    columns, results = execute_sql_query(final_sql_query, config)   # SQL 쿼리 실행 (데이터 조회)

    table_json = create_table_json(columns, results)
    chart_json = create_chart_json(columns, results)
    
    # 결과 출력
    if results:
        print("조회된 컬럼:\n", columns)
        for row in results:
            print(row)
        return table_json, chart_json
    else:
        print("조회 결과가 없습니다.")
        return None
    
    
##  수정중 =================================================================================================================================
"""
def first_llm(model, tokenizer, column_usage, user_query):
    PROMPT =\
    f'''
    <bos><start_of_turn>user
    너는 해운 회사의 데이터로 SQL 쿼리를 작성하는 데 도움을 주는 시스템이야.
    사용자로부터 받은 질문을 분석하여, 필요한 1) 필터 조건, 2) 집계 함수, 3) 정렬 기준, 4) SQL 쿼리, 5) SQL 쿼리에서 사용되는 컬럼 리스트를 제공해줘.  1~3은 필요하지 않은 경우에는 추출해주지 않아도 되고, 5는 4의 SQL 쿼리에서 사용된 모든 컬럼을 알려줘야 해.
    
    ### 참고 사항:
    1. 사용자 질문: 
    "{user_query}"
    
    2. 다음은 해운 회사 데이터의 메타데이터야. 테이블은 "revenue" 하나뿐이야:
    "{column_usage}"

    3. '최근'이라는 단어가 포함되는 경우, OUTBOR가 '2024-01-01' 이후
    
    ### 예시:
    질문: 최근 부산발 일본착 화주별 매출액을 분석해줘
    1) <filter/>OUTPOL = '부산', OUTPOD = '일본', OUTBOR = '2024-08-01 이후' <filter>
    2) <aggregation/>화주(고객)별 매출액의 합계<aggregation>
    3) <order/>매출액 합계 기준 내림차순<order>
    4) <sql_query/>SELECT OUTPOL,OUTPOD,OUTSHC,SUM(OUTSTL) AS TotalRevenue\n    FROM revenue\n    WHERE OUTPOL = \'한국\' AND OUTPOD = \'베트남\' AND OUTOBD >= \'20230101\' AND OUTOBD <= \'20230630\'\n    GROUP BY OUTSHC\n    ORDER BY TotalRevenue DESC;<sql_query>
    5) <columns/>OUTPOL,OUTPOD,OUTBOR,OUTSHC,OUTSTL<columns>

    <end_of_turn>
    <start_of_turn>model
    '''

    # Get Answer
    input_ids = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=2048)
    outputs_result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Regular expression to extract content between <query/> and <query>
    filter_pattern = r'<filter/>(.*?)<filter>'
    aggregation_pattern = r'<aggregation/>(.*?)<aggregation>'
    order_pattern = r'<order/>(.*?)<order>'
    sql_pattern = r'<sql_query/>(.*?)<sql_query>'
    columns_pattern = r'<columns/>(.*?)<columns>'
    
    filter_conditions = re.search(filter_pattern, outputs_result, re.DOTALL).group(1)
    aggregations = re.search(aggregation_pattern, outputs_result, re.DOTALL).group(1)
    orders = re.search(order_pattern, outputs_result, re.DOTALL).group(1)
    sql_queries = re.search(sql_pattern, outputs_result, re.DOTALL).group(1)
    parsed_columns = re.search(columns_pattern, outputs_result, re.DOTALL).group(1).split(",")

    return outputs_result, filter_conditions, aggregations, orders, sql_queries, parsed_columns

def extract_relevant_metadata(columns, metadata):
    relevant_metadata = {}
    for column in columns:
        if column in metadata["column_usage"]:
            relevant_metadata[column] = metadata["column_usage"][column]
    return relevant_metadata

def search_location_db(location, vector_db):
    # Vector DB에서 해당 위치에 대한 매핑 정보 검색
    return vector_db.get(location, "Mapping error")

def parse_and_augment_filter_conditions(filter_conditions, Metadata):
    pattern = r"(\w+)\s*=\s*'([^']+)'"  # ex: OUTPOL = '부산' 과 같은 패턴 추출
    matches = re.findall(pattern, filter_conditions)
    
    augmented_filters = []
    
    for col, val in matches:
        if col == 'OUTPOL' or col == 'OUTPOD':
            location_code = Metadata['location_code']
            mapped_value = search_location_db(val, location_code)
            if mapped_value != "UNKNOWN":
                augmented_filters.append(f"컬럼 {col}에 대한 값 '{val}' -> '{mapped_value}'로 매핑되었습니다.")
            else:
                augmented_filters.append(f"컬럼 {col}의 값 '{val}'에 대한 매핑 정보를 찾을 수 없습니다.")
    return "\n".join(augmented_filters)

outputs_1, filter_conditions, aggregations, orders, sql_queries, parsed_columns = first_llm(model, tokenizer, column_usage, query)
relevant_metadata = extract_relevant_metadata(parsed_columns, column_usage)  # 추출된 컬럼에 해당하는 메타데이터 가져오기
retrival_metadata = parse_and_augment_filter_conditions(filter_conditions, Metadata)   # Metadata와 매핑하여 구체화된 필터 조건 찾기

def second_llm(model, tokenizer, relevant_metadata, sql_queries, user_query, retrival_metadata):
    PROMPT =\
    f'''
    <bos><start_of_turn>user
    너는 해운 회사의 데이터로 정확한 SQL 쿼리를 작성해주는 시스템이야. 참고 사항을 이용해 SQL 쿼리 초안을 구체화해서, 사용자 질문에 답할 수 있는 정확한 SQL 쿼리를 만들고 이에 대해 설명해야 해. 

    ### 참고 사항:
    1. 다음은 너가 사용해야 할 데이터의 컬럼 메타데이터야: 
    "{relevant_metadata}"
    2. 너가 참고해야 할 정보야:
    {retrival_metadata}
    3. 예시는 출력 형식으로 참고만 하고, 꼭 SQL 쿼리 초안을 바탕으로 필요한 부분만 수정해서 정확한 SQL 쿼리를 만들어줘.

    ### SQL 쿼리 초안:
    {sql_queries}

    ### 사용자가 입력한 질문:
    "{user_query}"    

    ### 예시
    1) SQL_query: <sql_query/>SELECT OUTPOL,OUTPOD,OUTSHC,SUM(OUTSTL) AS TotalRevenue\n    FROM revenue\n    WHERE OUTPOL LIKE 'KRPUS' AND OUTPOD LIKE 'JP%' AND OUTOBD >= \'20230101\' AND OUTOBD <= \'20230630\'\n    GROUP BY OUTSHC\n    ORDER BY TotalRevenue DESC;<sql_query>
    2) 설명:
    
    
    <end_of_turn>
    <start_of_turn>model
    '''

    input_ids = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=2048)
    outputs_result = tokenizer.decode(outputs[0])

    sql_pattern = r'<sql_query/>(.*?)<sql_query>'
    sql_queries = re.search(sql_pattern, outputs_result, re.DOTALL).group(1)
    
    return outputs_result, sql_queries

outputs_2, final_sql_query = second_llm(model, tokenizer, relevant_metadata, sql_queries, query, retrival_metadata)
"""
##=================================================================================================================================
    
def first_llm(model, tokenizer, column_usage, user_query):
    PROMPT =\
    f'''
    <bos><start_of_turn>user
    너는 해운 회사의 데이터로 SQL 쿼리를 작성하는 데 도움을 주는 시스템이야. 사용자로부터 받은 질문을 분석하여, 필요한 컬럼과 필터 조건, 집계 함수 및 정렬 조건을 추출해줘.
    
    ### 참고 사항:
    1. 다음은 해운 회사 데이터의 메타데이터야. 테이블은 "revenue" 하나뿐이야.: 
    "{column_usage}"
    2. 사용자가 입력한 질문을 분석하여 필요한 컬럼을 식별하고, SQL 쿼리에서 사용할 필터 조건, 집계 함수, 정렬 기준을 제공해줘.
    
    ### 사용자가 입력한 질문:
    "{user_query}"
    
    ### 필요한 정보:
    1. 사용할 컬럼 (예: OUTDEM)
    2. 필터 조건 (예: 출발지로 한국, 최근 날짜는 2024-01-01 이후, 도착지는 몰타)
    3. 집계 함수 (필요한 경우, 예: 화주(고객)별 매출액의 합계)
    4. 정렬 조건 (필요한 경우, 예: 매출액 기준 내림차순)
    5. SQL 쿼리
    6. SQL 쿼리에 사용된 전체 컬럼 (예: OUTABS, OUTEDF)
    
    ### 출력 형식:
    1. 사용할 컬럼:
    2. 필터 조건:
    3. 집계 함수:
    4. 정렬 기준:
    5. SQL 쿼리: 
    6. SQL쿼리에 사용된 전체 컬럼:
    
    
    <end_of_turn>
    <start_of_turn>model
    '''
    
    input_text = PROMPT
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**input_ids, max_new_tokens=2048)
    outputs_result = tokenizer.decode(outputs[0])

    return outputs_result

def extract_sql_query(llm_output):
    ## 5. SQL 쿼리 Parsing 함수
    block_start = llm_output.find("<start_of_turn>model")        
    block_end = llm_output.find(";", block_start) + 1

    # 추출한 block 내에서 SELECT로 시작하는 쿼리 추출
    if block_start != -1 and block_end != -1:
        sql_block = llm_output[block_start + 6:block_end].strip()  # '''sql과 '''를 제외한 부분
        start_index = sql_block.find("SELECT") # "SELECT"로 시작하는 부분의 인덱스를 찾음
        end_index = sql_block.find(";", start_index) +1   # ";"로 끝나는 부분의 인덱스를 찾음 (; 미포함)

        # 인덱스 범위로 쿼리 부분 추출
        if start_index != -1 and end_index != -1:
            return sql_block[start_index:end_index]

    return None  # 쿼리를 찾지 못한 경우

def extract_columns_from_llm_output(llm_output):
    ## 6. SQL 쿼리에 사용한 전체 컬럼 Parsing 함수
    match = re.search(r'6\. \*\*SQL 쿼리에 사용된 전체 컬럼:\*\* (.*)', llm_output)    # "6. SQL 쿼리에 사용된 전체 컬럼" 이후 부분만 찾음
    if match:
        columns_str = match.group(1)  # 해당 부분 추출
        # OUT으로 시작하는 단어만 추출
        columns = re.findall(r'OUT\w+', columns_str)
        return columns
    return []

def extract_filter_conditions(llm_output):
    ## 2. 필터 조건 Parsing 함수
    start_index = llm_output.find("2. **필터 조건:**")    # "2. **필터 조건:**" 부분의 시작과 끝을 찾음
    end_index = llm_output.find("3. **집계 함수:**")
    
    # 필터 조건이 존재하는 경우 추출
    if start_index != -1 and end_index != -1:
        # "2. **필터 조건:**" 이후의 텍스트 부분만 추출
        filter_conditions = llm_output[start_index:end_index].strip()
        return filter_conditions
    else:
        return None  # 필터 조건을 찾지 못한 경우

# 검색 알고리즘 (Vector DB에서 매핑 정보 검색)
def search_location_db(location, vector_db):
    # Vector DB에서 해당 위치에 대한 매핑 정보 검색
    return vector_db.get(location, "Mapping error")

# Argumented prompt
def parse_and_augment_filter_conditions(filter_conditions, Metadata):
    pattern = r"(\w+)\s*=\s*'([^']+)'"  # ex: OUTPOL = '부산' 과 같은 패턴 추출
    matches = re.findall(pattern, filter_conditions)
    
    augmented_filters = []
    
    for col, val in matches:
        if col == 'OUTPOL' or col == 'OUTPOD':
            location_code = Metadata['location_code']
            mapped_value = search_location_db(val, location_code)
            if mapped_value != "UNKNOWN":
                augmented_filters.append(f"컬럼 {col}에 대한 값 '{val}' -> '{mapped_value}'로 매핑되었습니다.")
            else:
                augmented_filters.append(f"컬럼 {col}의 값 '{val}'에 대한 매핑 정보를 찾을 수 없습니다.")
    return "\n".join(augmented_filters)

# 추출된 컬럼에 해당하는 메타데이터만 가져오는 함수
def extract_relevant_metadata(columns, metadata):
    relevant_metadata = {}
    for column in columns:
        if column in metadata["column_usage"]:
            relevant_metadata[column] = metadata["column_usage"][column]
    return relevant_metadata

def execute_sql_query(sql_query, config):
    try:
        conn = sqlite3.connect(config.sql_data_path)        # SQLite 데이터베이스에 연결
        cursor = conn.cursor()

        if config.k is not None:
            sql_query = sql_query[:-1]
            sql_query += f"\nLIMIT {config.k};"
        else:
            sql_query += ";"
                 
        cursor.execute(sql_query)        # SQL 쿼리 실행
        
        result = cursor.fetchall()        # 결과 가져오기
        column_names = [description[0] for description in cursor.description]        # 컬럼 이름도 포함하기 위해 description을 사용

        cursor.close()
        conn.close()
        
        return column_names, result

    except sqlite3.Error as e:
        print(f"데이터베이스 오류 발생: {e}")
        return None, None

def second_llm(model, tokenizer, relevant_metadata, sql_query, retrival_metadata):
    PROMPT =\
    f'''
    <bos><start_of_turn>user
    너는 LLM이 추출한 컬럼과 조건을 메타데이터와 다시 매칭해, 보다 정확한 필터 조건을 만들어야 해. 다음 메타데이터를 참고해, 사용자 질문에서 추출된 조건을 구체화해주고, 기존 SQL 쿼리를 수정해줘.
    
    ### 참고 사항:
    1. 다음은 너가 사용해야 할 메타데이터와 정보야: 
    "{relevant_metadata}"
    2. 너가 수정해야 할 SQL 쿼리야
    {sql_query}
    3. 너가 참고해야 할 정보야.
    {retrival_metadata}
    
    ### 필요한 정보:
    1. 구체화된 필터 조건 (예: OUTPOL, OUTOBD, OUTPOD)
    2. 구체화된 필터 조건 (예: 출발지로 부산, 날짜는 2024년 이후, 도착지는 일본)
    3. 집계 함수 (필요한 경우, 예: 화주(고객)별 매출액의 합계)
    4. 정렬 조건 (필요한 경우, 예: 매출액 기준 내림차순)
    
    ### 출력 형식:
    - 구체화된 필터 조건:
    - 최종 SQL 쿼리:
    <end_of_turn>
    <start_of_turn>model
    '''

    input_text = PROMPT
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**input_ids, max_new_tokens=2048)
    outputs_result = tokenizer.decode(outputs[0])

    return outputs_result

def create_table_json(columns, results):
    head = "||".join(columns)    
    body = "^ ".join("||".join(map(str, row)) for row in results)    
    table_json = json.dumps({"head": head, "body": body})
    return table_json

def create_chart_json(columns, results):
    chart_data = [
        {"series": f"s{i+1}", "data": [{"x": str(row[0]), "y": str(row[1])}]}
        for i, row in enumerate(results)
    ]
    chart_json = json.dumps(chart_data)
    return chart_json