# 베이스 이미지 선택
FROM globeai/flux_ns:1.22

# 작업 디렉토리 설정
WORKDIR /workspace

# requirements.txt만 먼저 복사해서 종속성 설치 (캐시 활용)
COPY requirements.txt .

# pip 캐시 사용 안 함으로 설치 (임시 파일 최소화)
RUN pip install --no-cache-dir -r requirements.txt

# Solve the C compier
RUN apt-get update && apt-get install build-essential -y

# 현재 디렉토리의 모든 파일을 컨테이너의 /app 폴더로 복사
COPY . /workspace

# Flask 앱이 실행될 포트를 열어둠
EXPOSE 5000

# Ray Dashboard 포트 (8265)와 vLLM 관련 포트 필요 시 추가
EXPOSE 8265
# Expose port for the vLLM
EXPOSE 8000

# Flask 앱 실행 명령어
CMD ["python", "app.py"]
