"""
core 패키지 초기화 파일

Exports:
  [RAG.py]
    - execute_rag, execute_sql, generate_answer, query_sort, specific_question,
      sort_by_time, retrieve, expand_time_range_if_needed, cal_sim_score,
      cal_bm25_score, embed, min_max_scaling, generate, collect_vllm_text,
      generate_answer_stream, collect_vllm_text_stream

  [sql.py]
    - generate_sql (SQL 모듈 버전), first_llm, extract_relevant_metadata,
      parse_and_augment_filter_conditions, search_location_db, second_llm,
      execute_sql_query, create_table_json, create_chart_json

  [SQL_NS.py]
    - initialze, check_sqlplus, check_db_connection, get_all_schema_tables,
      make_metadata_from_table, run_sql_unno, get_metadata,
      generate_sql (SQL_NS 버전)
"""

from .RAG import (
    execute_rag,
    execute_sql,
    generate_answer,
    query_sort,
    specific_question,
    sort_by_time,
    retrieve,
    expand_time_range_if_needed,
    cal_sim_score,
    cal_bm25_score,
    embed,
    min_max_scaling,
    generate,
    collect_vllm_text,
    generate_answer_stream,
    collect_vllm_text_stream,
)

from .sql import (
    generate_sql as generate_sql_sql,
    first_llm,
    extract_relevant_metadata,
    parse_and_augment_filter_conditions,
    search_location_db,
    second_llm,
    execute_sql_query,
    create_table_json,
    create_chart_json,
)

from .SQL_NS import (
    initialze,
    check_sqlplus,
    check_db_connection,
    get_all_schema_tables,
    make_metadata_from_table,
    run_sql_unno,
    get_metadata,
    generate_sql as generate_sql_ns,
)

__all__ = [
    # RAG 모듈 관련 함수들
    "execute_rag",
    "execute_sql",
    "generate_answer",
    "query_sort",
    "specific_question",
    "sort_by_time",
    "retrieve",
    "expand_time_range_if_needed",
    "cal_sim_score",
    "cal_bm25_score",
    "embed",
    "min_max_scaling",
    "generate",
    "collect_vllm_text",
    "generate_answer_stream",
    "collect_vllm_text_stream",
    # sql 모듈 관련 함수들 (sql.py)
    "generate_sql_sql",
    "first_llm",
    "extract_relevant_metadata",
    "parse_and_augment_filter_conditions",
    "search_location_db",
    "second_llm",
    "execute_sql_query",
    "create_table_json",
    "create_chart_json",
    # SQL_NS 모듈 관련 함수들
    "initialze",
    "check_sqlplus",
    "check_db_connection",
    "get_all_schema_tables",
    "make_metadata_from_table",
    "run_sql_unno",
    "get_metadata",
    "generate_sql_ns",
]
