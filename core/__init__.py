"""
core 패키지 초기화 파일

Exports:
  [RAG.py]
    - execute_rag, generate_answer, query_sort, specific_question, generate_answer_stream
  
  [retrieval.py]
    - retrieve, sort_by_time, expand_time_range_if_needed, cal_sim_score, cal_bm25_score, 
      embed, min_max_scaling
  
  [generation.py]
    - generate, collect_vllm_text, collect_vllm_text_stream
    
  [image_processing.py]
    - image_query, image_streaming_query, prepare_image, prepare_multimodal_request
    
  [sql_processing.py]
    - generate_sql
    
  [sql.py]
    - first_llm, extract_relevant_metadata, parse_and_augment_filter_conditions,
      search_location_db, second_llm, execute_sql_query, create_table_json,
      create_chart_json

  [SQL_NS.py]
    - check_sqlplus, check_db_connection, get_all_schema_tables,
      make_metadata_from_table, run_sql_unno, get_metadata
"""

# RAG 모듈 - 코어 기능
from .RAG import (
    execute_rag,
    generate_answer,
    query_sort,
    specific_question,
    generate_answer_stream,
)

# Retrieval 모듈 - 검색 관련
from .retrieval import (
    retrieve,
    sort_by_time,
    expand_time_range_if_needed,
    cal_sim_score,
    cal_bm25_score,
    embed,
    min_max_scaling,
)

# Generation 모듈 - 생성 관련
from .generation import (
    generate,
    collect_vllm_text,
    collect_vllm_text_stream,
)

# Image Processing 모듈 - 이미지 처리 관련
from .image_processing import (
    image_query,
    image_streaming_query,
    prepare_image,
    prepare_multimodal_request,
)

# SQL Processing 모듈 - SQL 관련
from .sql_processing import (
    generate_sql,
)

# SQL 모듈 (원본 파일)
from .sql import (
    first_llm,
    extract_relevant_metadata,
    parse_and_augment_filter_conditions,
    search_location_db,
    second_llm,
    execute_sql_query,
    create_table_json,
    create_chart_json,
)

# SQL_NS 모듈 (원본 파일)
from .SQL_NS import (
    check_sqlplus,
    check_db_connection,
    get_all_schema_tables,
    make_metadata_from_table,
    run_sql_unno,
    get_metadata,
)

# prompt_test 모듈 - 새로운 프롬프트 테스트를 위한
from .prompt_test import (
    test_prompt_with_image,
    test_prompt_streaming,
)

# pdf_processor 모듈 - PDF 처리를 위해서
from .pdf_processor import (
    process_pdf,
    extract_images,
    extract_tables,
    pdf_to_prompt_context,
)


__all__ = [
    # RAG 모듈 관련 함수들
    "execute_rag",
    "generate_answer",
    "query_sort",
    "specific_question",
    "generate_answer_stream",

    # Retrieval 모듈 관련 함수들
    "retrieve",
    "sort_by_time",
    "expand_time_range_if_needed",
    "cal_sim_score",
    "cal_bm25_score",
    "embed",
    "min_max_scaling",

    # Generation 모듈 관련 함수들
    "generate",
    "collect_vllm_text",
    "collect_vllm_text_stream",

    # Image Processing 모듈 관련 함수들
    "image_query",
    "image_streaming_query",
    "prepare_image",
    "prepare_multimodal_request",

    # SQL Processing 모듈 관련 함수들
    "generate_sql",

    # sql 모듈 관련 함수들 (sql.py)
    "first_llm",
    "extract_relevant_metadata",
    "parse_and_augment_filter_conditions",
    "search_location_db",
    "second_llm",
    "execute_sql_query",
    "create_table_json",
    "create_chart_json",

    # SQL_NS 모듈 관련 함수들
    "check_sqlplus",
    "check_db_connection",
    "get_all_schema_tables",
    "make_metadata_from_table",
    "run_sql_unno",
    "get_metadata",
    
    # 새로운 프롬프트 테스트
    "test_prompt_with_image",
    "test_prompt_streaming",
    
    # PDF 처리 프로세서
    "process_pdf",
    "extract_images",
    "extract_tables",
    "pdf_to_prompt_context",
]