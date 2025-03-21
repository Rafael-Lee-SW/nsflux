# 베이스 이미지 선택
FROM globeai/flux_ns:1.26
# FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

# 2. 대화형 입력 없이 진행하도록 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive

# 작업 디렉토리 설정
WORKDIR /workspace


# Install necessary system packages and Python dependencies in one go (fewer layers)
# combining apt-get calls for efficiency
# Miniconda 설치에 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    vim \
    libaio1 \
    && rm -rf /var/lib/apt/lists/*

# # Miniconda (Python 3.11 버전용) 설치
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py311_23.3.1-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#     bash /tmp/miniconda.sh -b -p /opt/conda && \
#     rm /tmp/miniconda.sh

# # conda 설치 경로를 PATH에 추가
# ENV PATH=/opt/conda/bin:$PATH

# # Python 버전을 3.11.9로 업데이트하고, pip를 pip 24.0으로 업그레이드
# RUN conda install python=3.11.9 -y && \
#     pip install --upgrade pip==24.0

# # 버전 확인 (빌드 시 로그에 표시됨)
# RUN python --version && pip --version

# (4) CUDA_HOME 환경변수 설정
#     which nvcc 결과가 /usr/bin/nvcc 이고 실제 툴킷이 /usr/lib/nvidia-cuda-toolkit 내에 있으므로 해당 경로를 지정
# ENV CUDA_HOME="/usr/lib/nvidia-cuda-toolkit"
# ENV PATH="${CUDA_HOME}/bin:${PATH}"
# ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# # Solve the C compier
# RUN apt-get update && apt-get install build-essential -y

# requirements.txt만 먼저 복사해서 종속성 설치 (캐시 활용)
COPY requirements.txt .

# pip 캐시 사용 안 함으로 설치 (임시 파일 최소화)
RUN pip install --no-cache-dir -r requirements.txt

# gemma3.py 패치 파일 복사 (vLLM gemma3 파일 덮어쓰기)
# COPY patches/vllm/gemma3.py /opt/conda/lib/python3.11/site-packages/vllm/model_executor/models/gemma3.py

# Additional pip installations:
# RUN pip install ninja \
#     && pip install git+https://github.com/huggingface/transformers.git \
#     && pip install git+https://github.com/vllm-project/vllm.git

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
