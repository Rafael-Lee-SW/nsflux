# 베이스 이미지 선택
FROM globeai/flux_ns:env

# 작업 디렉토리 설정
WORKDIR /workspace

# Copy files and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 현재 디렉토리의 모든 파일을 컨테이너의 /app 폴더로 복사
COPY . /workspace

# Flask 앱이 실행될 포트를 열어둠
EXPOSE 5000

# Flask 앱 실행 명령어
CMD ["python", "app.py"]
