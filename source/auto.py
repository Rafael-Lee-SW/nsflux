#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
프로젝트 내에서 미리 정의된 파일 목록만을 대상으로 템플릿 파일을 자동 생성하는 스크립트.
생성된 결과 파일은 auto/source_code.txt 로 저장됨.
"""

import os
import logging
import chardet  # pip install chardet

# 로깅 설정 (시간, 레벨, 메시지 출력)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 미리 선택한 파일 목록 (상대경로)
ALLOWED_FILES = [
    # ".dockerignore",
    "config.yaml",
    "Dockerfile",
    "requirements.txt",
    "app.py",
    "ray_deploy/ray_setup.py",
    "ray_deploy/ray_utils.py",
    # "ray_deploy/langchain.py",
    # "ray_deploy/sse_manager.py",
    # "utils/utils_format.py",
    "utils/utils_load.py",
    # "utils/utils_vector.py",
    "utils/debug_tracking.py",
    # "utils/summarizer.py",
    # "utils/tracking.py",
    "core/generation.py",
    "core/image_processing.py",
    "core/RAG.py",
    "core/retrieval.py",
    "core/SQL_NS.py",  # 검증 대상 파일 포함
    # "core/sql_processing.py",
    # "prompt/prompt_rag.py",
    # "prompt/prompt_sql.py",
    # "data_control/data_control.py",
    # "templates/data_manager.html",
    # "templates/chatroom.html",
    # "templates/index_test.html",
    # "vectorize.ipynb"
]

def get_code_block_language(file_path):
    """
    파일명 또는 확장자에 따라 적절한 코드 블록 언어를 반환.
    해당 언어가 없으면 빈 문자열을 반환하여, 일반 코드 블록으로 출력함.
    """
    base = os.path.basename(file_path)
    # 특정 파일명에 대한 처리
    if base == "Dockerfile":
        return "dockerfile"
    if base == ".dockerignore":
        return ""  # 별도 언어 없음

    # 확장자에 따른 처리
    _, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext == ".py":
        return "python"
    elif ext in [".yaml", ".yml"]:
        return "yaml"
    elif ext == ".txt":
        return "txt"
    elif ext == ".html":
        return "html"
    else:
        return ""

def build_tree_from_paths(paths):
    """
    주어진 경로 목록으로부터 트리 구조(중첩 딕셔너리)를 생성.
    """
    tree = {}
    for path in paths:
        parts = path.split(os.sep)
        current = tree
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
    return tree

def tree_to_string(tree, prefix=""):
    """
    중첩 딕셔너리 형태의 트리 구조를 tree 커넥터(├─, └─ 등)를 사용한 문자열 리스트로 반환.
    """
    lines = []
    keys = sorted(tree.keys())
    for i, key in enumerate(keys):
        connector = "└─ " if i == len(keys) - 1 else "├─ "
        lines.append(prefix + connector + key)
        if tree[key]:
            extension = "    " if i == len(keys) - 1 else "│   "
            lines.extend(tree_to_string(tree[key], prefix + extension))
    return lines

def read_file_with_encoding(file_path):
    """
    chardet를 사용하여 파일의 인코딩을 감지한 후 해당 인코딩으로 파일을 읽어 반환하는 함수.
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] if result['encoding'] else 'utf-8'
        # errors 옵션을 추가하여 디코딩 오류 발생 시 대체 처리
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            return f.read()
    except Exception as e:
        logging.error(f"파일 읽기 오류: {file_path} - {str(e)}")
        return f"# 파일 읽기 오류: {str(e)}"

def main():
    try:
        base_dir = os.getcwd()  # 프로젝트 루트 디렉토리
        
        # ALLOWED_FILES 목록 기반으로 트리 구조 생성
        tree_dict = build_tree_from_paths(ALLOWED_FILES)
        tree_lines = tree_to_string(tree_dict)
        tree_str = "\n".join(tree_lines)

        output_lines = []
        # 헤더에 프로젝트 트리 출력
        output_lines.append("# Project Tree of RAG for company\n")
        output_lines.append("```\n" + tree_str + "\n```\n")

        # --- SQL_NS.py 검증 ---
        sql_ns_file = "core/SQL_NS.py"
        sql_ns_full_path = os.path.join(base_dir, sql_ns_file)
        try:
            content_sql_ns = read_file_with_encoding(sql_ns_full_path)
            logging.info(f"{sql_ns_file} 파일 검증 완료 (문자열 길이: {len(content_sql_ns)})")
        except Exception as e:
            logging.error(f"{sql_ns_file} 파일 검증 실패: {str(e)}")

        # 각 파일의 소스 코드를 코드 블록으로 추가 (ALLOWED_FILES 순서대로)
        for file_path in ALLOWED_FILES:
            output_lines.append(f"--- {file_path}\n")
            lang = get_code_block_language(file_path)
            if lang:
                output_lines.append(f"```{lang}\n")
            else:
                output_lines.append("```\n")
            full_file_path = os.path.join(base_dir, file_path)
            if os.path.isfile(full_file_path):
                content = read_file_with_encoding(full_file_path)
            else:
                logging.error(f"파일이 존재하지 않음: {file_path}")
                content = "# 파일이 존재하지 않습니다."
            output_lines.append(content.rstrip() + "\n")
            output_lines.append("```\n\n")

        # Base-Knowledge와 My-Requirements 두 파트로 요구사항 섹션 구성
        requirements_text = (
            "# Base-Knowledge\n\n"
            " - 위 파일들은 LLM 모델을 활용한 사내 RAG 서비스의 소스 코드입니다.\n"
            " - 파일 트리와 각 파일의 내용이 코드 블록 내에 포함되어, 프로젝트의 현재 구조와 상태를 한눈에 파악할 수 있습니다.\n"
            " - vLLM과 ray를 활용하여 사용성 및 추론 성능을 개선하였습니다.\n"
            " - Langchain을 활용하여 reqeust_id별로 대화를 저장하고 활용할 수 있습니다.\n"
            " - 에러 발생 시 로깅을 통해 문제를 추적할 수 있도록 설계되었습니다.\n\n\n"
            "# Answer-Rule\n\n"
            " 1. 추후 소스 코드 개선, 구조 변경, 에러 로그 추가 등 다양한 요구사항을 반영할 수 있는 확장성을 고려합니다.\n"
            " 2. 전체 코드는 한국어로 주석 및 설명이 포함되어, 이해와 유지보수가 용이하도록 작성됩니다.\n\n\n"
            "# My-Requirements\n\n"
            " 1. User requirements.\n"
            " 2. My requirements.\n"
        )
        output_lines.append("-----------------\n")
        output_lines.append(requirements_text)

        final_output = "\n".join(output_lines)

        # 'auto' 폴더 생성 (없으면 생성)
        source_dir = os.path.join(base_dir, "source")
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
            logging.info(f"'source' 폴더 생성됨: {source_dir}")

        # 결과 파일 이름 (버전 및 날짜 포함)
        output_file_name = "source_code.txt"
        output_file_path = os.path.join(source_dir, output_file_name)
        with open(output_file_path, "w", encoding="utf-8") as out_file:
            out_file.write(final_output)

        logging.info(f"결과 파일 생성 완료: {output_file_path}")

    except Exception as e:
        logging.error(f"메인 실행 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
