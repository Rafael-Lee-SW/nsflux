# core/SQL_NS.py

import os
import subprocess
from utils.tracking import time_tracker
import json
import yaml
from box import Box
import re

# 환경 변수 설정
os.environ['ORACLE_HOME'] = '/workspace/oracle/instantclient_23_7'
os.environ['LD_LIBRARY_PATH'] = os.environ['ORACLE_HOME'] + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PATH'] = os.environ['ORACLE_HOME'] + ':' + os.environ.get('PATH', '')

# Config 불러오기 (DB 접속 등 기타 정보를 config.yaml에서 가져온다고 가정)
with open("./config.yaml", "r") as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)

# 기본 SQL 접속코드 (※ 실제 DB 접속 계정/주소는 본인 환경에 맞게 수정 필요)
sqlplus_command = [
    "sqlplus", "-S", "LLM/L9SD2TT9XJ0H@//210.113.16.230:1521/ORA11GDR"
]

'''
### ORACLE DB 정보 ###
TABLE : ai_dg_check
    COLUMNS : CLS (위험물 클래스)
              UNNO (위험물 UN 번호)
              PORT (포트 번호)
              ALLOW_YN (취급 가능 여부)
'''

@time_tracker
def check_sqlplus():
    """
    sqlplus 버전 정보 확인
    """
    try:
        result = subprocess.run(['sqlplus', '-version'], capture_output=True, text=True, check=True)
        print(" SQL*Plus is working!")
        print("Version info:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")


@time_tracker
def check_db_connection():
    """
    DB 연결이 정상적인지 테스트
    """
    try:
        sql_query = "SELECT 1 FROM dual;\nEXIT;\n"
        result = subprocess.run(
            sqlplus_command,
            input=sql_query,
            capture_output=True,
            text=True
        )

        if "1" in result.stdout:
            print("  Successfully connected to the Namsung database!")
        else:
            print(" Connection to the database failed!")

    except subprocess.CalledProcessError as e:
        print(f" Error: {e.stderr}")


@time_tracker
def get_all_schema_tables():
    """
    모든 스키마, 테이블 목록 조회
    """
    try:
        sqlplus_cmd = [
            'sqlplus', '-S', 'LLM/L9SD2TT9XJ0H@//210.113.16.230:1521/ORA11GDR'
        ]
        sql_query = """SET PAGESIZE 0 FEEDBACK OFF VERIFY OFF HEADING OFF ECHO OFF;
        SELECT OWNER, TABLE_NAME FROM ALL_TABLES ORDER BY OWNER, TABLE_NAME;
        EXIT;"""

        result = subprocess.run(
            sqlplus_cmd,
            input=sql_query,
            capture_output=True,
            text=True
        )

        schema_tables = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    schema, table = parts[0], parts[1]
                    if schema not in schema_tables:
                        schema_tables[schema] = []
                    schema_tables[schema].append(table)

        if schema_tables:
            print("  스키마별 테이블 목록:")
            for schema, tables in schema_tables.items():
                print(f"\n🔹 스키마: {schema}")
                for t in tables:
                    print(f"  - {t}")
        else:
            print(" 테이블이 존재하지 않습니다.")

        return schema_tables

    except subprocess.CalledProcessError as e:
        print(f" Error: {e.stderr}")
        return {}


def make_metadata_from_table(schema_name="ICON", table_name="OPRAIMDG"):
    """
    예시 함수: OPRAIMDG 테이블로부터 UN, CLASS, DESCRIPTION 정보를 JSON으로 만드는 함수
    """
    sql_query = f"""
    SET LINESIZE 2000;
    SET PAGESIZE 0;
    SET TRIMSPOOL ON;
    COL IMDCOM FORMAT A200;
    SELECT IMDUNM, IMDCLS, REPLACE(REPLACE(IMDCOM, CHR(10), ' '), CHR(13), ' ') AS IMDCOM 
    FROM {schema_name}.{table_name};
    EXIT;
    """

    try:
        result = subprocess.run(sqlplus_command, input=sql_query, capture_output=True, text=True)
        print(f"  RESULT: \n{str(result)[:1000]}")
        output = result.stdout
        print(f"  OUTPUT: \n{str(output)[:1000]}")

        lines = output.strip().split("\n")
        print(f"  LINE: \n{str(lines)[:1000]}")
        metadata = []

        for line in lines[:-1]:
            values = line.split(None, 2)
            if len(values) == 3:
                imdunm = values[0].strip()
                imdcls = values[1].strip()
                imdcom = values[2].strip()
                metadata.append({
                    "UNNO": imdunm,
                    "Class": imdcls,
                    "Description": imdcom
                })

        json_filename = "/workspace/data/METADATA_OPRAIMDG.json"
        with open(json_filename, "w", encoding="utf-8") as json_file:
            json.dump(metadata, json_file, indent=4, ensure_ascii=False)

        print(f"  Metadata saved to {json_filename}")

    except subprocess.CalledProcessError as e:
        print(f" SQL Execution Error: {e.stderr}")


@time_tracker
def run_sql_unno(cls=None, unno=None, pol_port='KR%', pod_port='JP%'):
    """
    ai_dg_check 테이블에서 CLS, UNNO, PORT에 대한 DG 선적 가능 여부 조회
    """
    cls_val = "NULL" if (cls is None or cls == "NULL") else f"'{cls}'"
    unno_val = "NULL" if (unno is None or unno == "NULL") else f"'{unno}'"

    sql_query = f"""
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
    WHERE (p.cls={cls_val} OR {cls_val} IS NULL) AND (p.unno={unno_val} OR {unno_val} IS NULL) 
      AND p.port LIKE '{pol_port}'
      AND (p.cls={cls_val} OR {cls_val} IS NULL) AND (d.unno={unno_val} OR {unno_val} IS NULL) 
      AND d.port LIKE '{pod_port}';
    EXIT;
    """

    try:
        result = subprocess.run(sqlplus_command, input=sql_query, capture_output=True, text=True)
        print("  SQL Query Results:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f" Error: {e.stderr}")

    return sql_query, result.stdout


@time_tracker
def run_sql_bl(cls=None, unno=None, pol_port='KR%', pod_port='JP%'):
    """
    B/L 상세 조회
    """
    cls_val = "NULL" if (cls is None or cls == "NULL") else f"'{cls}'"
    unno_val = "NULL" if (unno is None or unno == "NULL") else f"'{unno}'"

    sql_query = f"""
    SELECT *
    FROM (
        SELECT
            MST.FRTBNO AS "B/L No",
            MST.FRTOBD AS onBoard_Date,
            MST.FRTPOL AS POL,
            MST.FRTPOD AS POD,
            MST.FRTSBM AS ship_back,
            CNT.KCTUNN AS UNNO,
            CNT.KCTCLS AS CLASS,
            COUNT(*) AS "DG_Container_Count"
        FROM ICON.WSDAMST MST
        JOIN ICON.WSDACNT CNT ON CNT.KCTBNO = MST.FRTBNO
        WHERE MST.BUKRS = '1000'
        AND CNT.BUKRS = '1000'
        AND MST.FRTOBD BETWEEN TO_CHAR(SYSDATE-1095,'YYYYMMDD')+1 AND TO_CHAR(SYSDATE+1,'YYYYMMDD')
        AND CNT.KCTUNN = {unno_val}
        AND CNT.KCTCLS = {cls_val}
        AND MST.FRTPOL = '{pol_port}'
        AND MST.FRTPOD = '{pod_port}'
        GROUP BY
            MST.FRTBNO,
            MST.FRTOBD,
            MST.FRTPOL,
            MST.FRTPOD,
            MST.FRTSBM,
            CNT.KCTUNN,
            CNT.KCTCLS
    )
    WHERE ROWNUM <= 5;
    EXIT;
    """

    try:
        result = subprocess.run(sqlplus_command, input=sql_query, capture_output=True, text=True)
        print("[SQL_NS] SQL Query run_sql_bl Results:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[SQL_NS] run_sql_bl Error: {e.stderr}")

    return sql_query, result.stdout


def get_metadata(config):
    """
    metadata_path (port 데이터), metadata_unno (UN 번호 리스트)에 있는 내용을 불러와 반환
    """
    print("[SOOWAN] get_metadata 진입")
    if not config or not hasattr(config, "metadata_unno"):
        raise ValueError("Config 객체에 'metadata_unno' 속성이 없습니다.")

    unno_path = config.metadata_unno
    port_path = config.metadata_path

    with open(port_path, "r", encoding="utf-8") as f:
        port_data = json.load(f)
    location_codes = json.dumps(port_data.get("location_code"), ensure_ascii=False)

    with open(unno_path, "r", encoding="utf-8") as f:
        unno_data = json.load(f)
    unno_list_as_string = json.dumps(unno_data, ensure_ascii=False)

    return location_codes, unno_list_as_string


if __name__ == "__main__":
    # 아래는 테스트/디버깅용 코드
    # 필요한 경우에만 사용 가능. 실제 운영 시엔 제거할 수도 있음.

    check_sqlplus()             # sqlplus가 잘 동작하는지 확인
    check_db_connection()       # 데이터베이스 접속 여부 확인
    schema_info = get_all_schema_tables()
    print("Schema info:", schema_info)
    # make_metadata_from_table()  # 특정 테이블로부터 메타데이터 생성하는 예시
    # 예시 SQL 실행
    sql_q, sql_res = run_sql_unno(cls=4.1, unno=1033, pol_port="KRPUS", pod_port="JPKOB")
    print("[TEST] run_sql_unno result:", sql_q, sql_res)
    sql_q2, sql_res2 = run_sql_bl(cls=4.1, unno=1033, pol_port="KRPUS", pod_port="JPKOB")
    print("[TEST] run_sql_bl result:", sql_q2, sql_res2)
