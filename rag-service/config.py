from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # 모델 설정
    MODEL_NAME: str = "Linq-AI-Research/Linq-Embed-Mistral"
    USE_ONNX: bool = False
    ONNX_MODEL_PATH: str = "./models/onnx"
    
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # 데이터 설정
    DATA_PATH: str = "./data/0404_Mistral_DB.json"
    IMAGE_BASE_PATH: str = '/globeai/IMAGE/result_3' # 이미지 경로
    MIN_DOCS: int = 50
    
    # 캐시 설정
    CACHE_DIR: str = "cache"
    HF_TOKEN_PATH: str = "/root/.cache/huggingface/token"
    
    # 멀티프로세싱 설정
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 32
    
    # Ray 설정
    USE_RAY: bool = False  # Ray 사용 여부
    RAY_ADDRESS: str = "local"  # 로컬 모드로 변경
    RAY_NUM_CPUS: int = 4
    RAY_NUM_GPUS: int = 0
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# 캐시 디렉토리 생성
os.makedirs(settings.CACHE_DIR, exist_ok=True)