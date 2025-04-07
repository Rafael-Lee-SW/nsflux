# RAG 서비스

이 서비스는 Linq-AI-Research/Linq-Embed-Mistral 모델을 사용하여 문서 검색 기능을 제공합니다.

## 기능

- 문서 임베딩 및 검색
- 시간 기반 필터링
- 유사도 및 BM25 기반 스코어링
- FastAPI 기반 REST API

## 설치 및 실행

### 로컬 환경에서 실행

1. 의존성 설치:
```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:
```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 설정을 변경
```

3. 서버 실행:
```bash
uvicorn app:app --reload
```

### Docker를 사용하여 실행

1. 이미지 빌드:
```bash
docker build -t rag-service .
```

2. 컨테이너 실행:
```bash
docker run -p 8000:8000 -v $(pwd)/data:/app/data rag-service
```

## API 엔드포인트

- `POST /api/retrieve`: 문서 검색 API
  - 요청 본문:
    ```json
    {
      "query": "검색할 쿼리",
      "top_n": 5,
      "time_range": "all"  // 또는 "2020-01-01:2023-12-31" 형식
    }
    ```
  - 응답:
    ```json
    {
      "documents": "검색된 문서 내용",
      "documents_list": [
        {
          "file_name": "파일명",
          "title": "제목",
          "contents": "내용",
          "chunk_id": "청크 ID"
        },
        ...
      ]
    }
    ```

- `GET /api/health`: 서비스 상태 확인

## 설정

환경 변수를 통해 다음 설정을 변경할 수 있습니다:

- `PORT`: 서버 포트 (기본값: 8000)
- `HOST`: 서버 호스트 (기본값: 0.0.0.0)
- `EMBED_MODEL_ID`: 임베딩 모델 ID
- `CACHE_DIR`: 모델 캐시 디렉토리
- `DATA_PATH`: 벡터 데이터베이스 파일 경로
- `MIN_DOCS`: 최소 필요 문서 수
- `LOG_LEVEL`: 로깅 레벨
