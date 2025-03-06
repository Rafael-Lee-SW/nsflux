import os
import subprocess

# 환경 변수 설정
os.environ['ORACLE_HOME'] = '/workspace/sql/instantclient_23_7'
os.environ['LD_LIBRARY_PATH'] = os.environ['ORACLE_HOME'] + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PATH'] = os.environ['ORACLE_HOME'] + ':' + os.environ.get('PATH', '')

# sqlplus 실행 여부 확인
def check_sqlplus():
    try:
        # sqlplus 버전 확인
        result = subprocess.run(['sqlplus', '-version'], capture_output=True, text=True, check=True)
        print("SQL*Plus is working!")
        print("Version info:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")

# DB 접속 확인
def check_db_connection():
    try:
        # SQL*Plus를 실행하여 데이터베이스에 접속 가능한지 테스트
        test_command = [
            'sqlplus', '-S', 'LLM/L9SD2TT9XJ0H@//210.113.16.230:1521/ORA11GDR', 
            'SELECT 1 FROM dual;'
        ]
        result = subprocess.run(test_command, capture_output=True, text=True, check=True)
        
        # SQL*Plus 결과 출력
        if "1" in result.stdout:
            print("Successfully connected to the Namsung database!")
        else:
            print("Connection to the database failed!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")

# Oracle sqlplus 명령어 실행 예시
def run_sqlplus_query():
    # SQL*Plus 명령어를 실행할 기본 명령어
    sqlplus_command = [
        'sqlplus',  # sqlplus 실행 파일
        '-S',  # 간단한 출력 모드
        'LLM/L9SD2TT9XJ0H@//210.113.16.230:1521/ORA11GDR',  # Oracle 접속 정보
        '@', '/workspace/sql/DG_check.sql'  # SQL 파일 실행 (예: DG_CHECK1.sql)
    ]
    
    # subprocess를 사용하여 SQL*Plus 명령어 실행
    try:
        result = subprocess.run(sqlplus_command, capture_output=True, text=True, check=True)
        # SQL*Plus의 출력 결과를 받아옵니다
        print("SQL Query Results:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        # 오류가 발생한 경우 오류 메시지 출력
        print(f"Error: {e.stderr}")

if __name__ == "__main__":
    check_sqlplus()  # sqlplus가 잘 동작하는지 확인
    check_db_connection()  # 데이터베이스 접속 여부 확인
    run_sqlplus_query()  # 실제 SQL 쿼리 실행
