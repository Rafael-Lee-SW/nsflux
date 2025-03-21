"""
prompts 패키지 초기화 파일

Exports:
    - QUERY_SORT_PROMPT: 질문 유형 분류 및 구체화 프롬프트 템플릿.
    - QUERY_SORT_PROMPT_OLD: 이전 버전의 질문 유형 분류 프롬프트 템플릿.
    - GENERATE_PROMPT_TEMPLATE: 내부 자료 기반 답변 생성 프롬프트 템플릿.
    - STREAM_PROMPT_TEMPLATE: 스트리밍 방식 답변 생성 프롬프트 템플릿.
"""

from .prompt_rag import (
    QUERY_SORT_PROMPT,
    QUERY_SORT_PROMPT_OLD,
    GENERATE_PROMPT_TEMPLATE,
    STREAM_PROMPT_TEMPLATE,
)

from .prompt_sql import (
    SQL_EXTRACTION_PROMPT_TEMPLATE
)

__all__ = [
    "QUERY_SORT_PROMPT",
    "QUERY_SORT_PROMPT_OLD",
    "GENERATE_PROMPT_TEMPLATE",
    "STREAM_PROMPT_TEMPLATE",
    "SQL_EXTRACTION_PROMPT_TEMPLATE"
]
