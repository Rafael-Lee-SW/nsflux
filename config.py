import yaml
from box import Box
import logging

def load_config():
    """설정을 로드하고 Box 객체로 반환합니다."""
    with open("./config.yaml", "r") as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config = Box(config_yaml)
    
    # RAG 서비스 URL 설정이 없는 경우 기본값 설정
    if not hasattr(config, 'rag_service_url'):
        config.rag_service_url = "http://localhost:8000"
        logging.info(f"RAG 서비스 URL이 설정되지 않았습니다. 기본값 '{config.rag_service_url}'을 사용합니다.")
    else:
        logging.info(f"RAG 서비스 URL: {config.rag_service_url}")
    
    return config

# 전역 설정 객체
config = load_config() 