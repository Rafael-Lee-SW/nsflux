# utils/utils_format.py
import json
import numpy as np
import torch
import random
import shutil
from datetime import datetime, timedelta
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
)

import os
import requests

# Import vLLM utilities
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# Define the minimum valid file size (e.g., 10MB)
MIN_WEIGHT_SIZE = 10 * 1024 * 1024

# For tracking execution time of functions
from utils.tracking import time_tracker

# Logging
import logging

logging.basicConfig(level=logging.DEBUG)


# -------------------------------------------------
# Function: process_to_format
# -------------------------------------------------
@time_tracker
def process_to_format(qry_contents, type):
    # 여기서 RAG 시스템을 호출하거나 답변을 생성하도록 구현하세요.
    # 예제 응답 형식
    ### rsp_type : RA(Retrieval All), RT(Retrieval Text), RB(Retrieval taBle), AT(Answer Text), AB(Answer taBle) ###
    print("[SOOWAN] process_to_format 진입")
    if type == "Retrieval":
        print("[SOOWAN] 타입 : 리트리버")
        tmp_format = {"rsp_type": "R", "rsp_tit": "남성 내부 데이터", "rsp_data": []}
        for i, form in enumerate(qry_contents):
            tmp_format_ = {
                "rsp_tit": f"{i+1}번째 검색데이터: {form['title']} (출처:{form['file_name']})",
                "rsp_data": form["contents"],
                "chunk_id": form.get("chunk_id"),
            }
            tmp_format["rsp_data"].append(tmp_format_)
        return tmp_format

    elif type == "SQL":
        print("[SOOWAN] 타입 : SQL")
        tmp_format = {
            "rsp_type": "R",
            "rsp_tit": "남성 내부 데이터",
            "rsp_data": [{"rsp_tit": "SQL Query 결과표", "rsp_data": []}],
        }
        tmp_format_sql = {
            "rsp_type": "TB",
            "rsp_tit": qry_contents[0]["title"],
            "rsp_data": qry_contents[0]["data"],
        }
        tmp_format_chart = {
            "rsp_type": "CT",
            "rsp_tit": qry_contents[1]["title"],
            "rsp_data": {"chart_tp": "BAR", "chart_data": qry_contents[1]["data"]},
        }
        tmp_format["rsp_data"][0]["rsp_data"].append(tmp_format_sql)
        # tmp_format['rsp_data'].append(tmp_format_chart)
        return tmp_format, tmp_format_chart

    elif type == "Answer":
        print("[SOOWAN] 타입 : 대답")
        tmp_format = {"rsp_type": "A", "rsp_tit": "답변", "rsp_data": []}
        # for i, form in enumerate(qry_contents):
            # if i == 0:
        tmp_format_ = {"rsp_type": "TT", "rsp_data": qry_contents}
        tmp_format["rsp_data"].append(tmp_format_)
            # elif i == 1:
            #     tmp_format["rsp_data"].append(form)
            # else:
            #     None

        return tmp_format

    else:
        print("Error! Type Not supported!")
        return None

# @time_tracker
# def process_format_to_response(formats, qry_id, continue_="C", update_index=1):
#     # Get multiple formats to tuple

#     ans_format = {
#         "status_code": 200,
#         "result": "OK",
#         "detail": "",
#         "continue":continue_,
#         "qry_id": qry_id,
#         "rsp_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
#         "data_list": [],
#     }

#     # 누적된 토큰을 하나의 문자열로 결합합니다.
#     aggregated_answer = "".join(token.get("answer", "") for token in formats)
#     ans_format["data_list"].append({
#         "rsp_type": "A",
#         "rsp_tit": f"답변{update_index}",
#         "rsp_data": [
#             {
#                 "rsp_type": "TT",
#                 "rsp_data": aggregated_answer
#             }
#         ]
#     })
    
#     # Validate JSON before returning
#     try:
#         json.dumps(ans_format, ensure_ascii=False)  # Test JSON validity
#     except Exception as e:
#         print(f"[ERROR] Invalid JSON structure: {str(e)}")
#         ans_format["status_code"] = 500
#         ans_format["result"] = "ERROR"
#         ans_format["detail"] = f"JSON Error: {str(e)}"

#     # for format in formats:
#     #     ans_format["data_list"].append(format)

#     # return json.dumps(ans_format, ensure_ascii=False)
#     return ans_format

@time_tracker
def process_format_to_response(formats, qry_id, continue_="C", update_index=1):
    # If there are any reference tokens, return only them.
    reference_tokens = [token for token in formats if token.get("type") == "reference"]
    if reference_tokens:
        # For this example, we'll use the first reference token.
        ref = reference_tokens[0]
        # Add the extra keys.
        ref["qry_id"] = qry_id
        ref["continue"] = continue_
        ref["rsp_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        # Ensure that a "rsp_tit" key exists to satisfy downstream requirements.
        if "rsp_tit" not in ref:
            ref["rsp_tit"] = "Reference"
        return ref

    # Otherwise, aggregate the normal answer tokens.
    normal_tokens = [token.get("answer", "") for token in formats if token.get("type") != "reference"]
    aggregated_answer = "".join(normal_tokens)
    
    ans_format = {
        "status_code": 200,
        "result": "OK",
        "detail": "",
        "continue": continue_,
        "qry_id": qry_id,
        "rsp_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "data_list": [{
            "rsp_type": "A",
            "rsp_tit": f"답변{update_index}",
            "rsp_data": [{
                "rsp_type": "TT",
                "rsp_data": aggregated_answer
            }]
        }]
    }
    
    # Validate JSON structure before returning.
    try:
        json.dumps(ans_format, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] Invalid JSON structure: {str(e)}")
        ans_format["status_code"] = 500
        ans_format["result"] = "ERROR"
        ans_format["detail"] = f"JSON Error: {str(e)}"
    
    return ans_format



# @time_tracker
# def process_format_to_response(formats, qry_id, continue_="C", update_index=1):
#     # 누적된 토큰들을 하나의 문자열로 결합합니다.
#     aggregated_answer = "".join(token.get("answer", "") for token in formats)
    
#     # retrieval과 동일한 구조를 위해, 답변 데이터는 내부 data_list가 딕셔너리 형태로 구성됩니다.
#     answer = {
#         "rsp_type": "A",                # Answer
#         "rsp_tit": f"답변{update_index}",
#         "rsp_data": [                    # 바로 텍스트 응답 리스트를 구성
#             {
#                 "rsp_tit": f"답변{update_index}",
#                 "rsp_data": [
#                     {
#                         'rsp_type': 'TT',
#                         'rsp_tit': '',
#                         'rsp_data': aggregated_answer,
#                     }
#                 ]
                
#             }
#         ]
#     }
    
#     # 최종 응답 구조: 최상위에 data_list는 리스트이고, 내부에 딕셔너리로 답변 데이터를 포함합니다.
#     ans_format = {
#         "status_code": 200,
#         "result": "OK",
#         "detail": "Answer",
#         "continue": continue_,
#         "qry_id": qry_id,
#         "rsp_time": datetime.now().isoformat(),
#         "data_list": [
#             {
#                 "type": "answer",               # 응답 타입 answer
#                 "status_code": 200,
#                 "result": "OK",
#                 "detail": "Answer",
#                 "evt_time": datetime.now().isoformat(),
#                 "data_list": answer              # retrieval의 data_list와 동일하게 딕셔너리 형태
#             }
#         ]
#     }
#     return ans_format

@time_tracker
def error_format(message, status, qry_id=""):
    ans_format = {
        "status_code": status,
        "result": message,
        "qry_id": qry_id,  # 추가: qry_id 포함
        "detail": "",
        "evt_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
    }
    return json.dumps(ans_format)

# @time_tracker
# def send_data_to_server(data, url):
#     headers = {
#         "Content-Type": "application/json; charset=utf-8"
#     }
#     try:
#         # 다른 서버로 데이터를 전송 (POST 요청)
#         response = requests.post(url, json=data, headers=headers)
#         if response.status_code == 200:
#             print(f"Data sent successfully: {data}")
#         else:
#             print(f"Failed to send data: {response.status_code}")
#             print(f"Failed data: {data}")
#     except requests.exceptions.RequestException as e:
#         print(f"Error sending data: {e}")
@time_tracker     
def send_data_to_server(data, url):
    try:
        if not data or "data_list" not in data:
            print("[ERROR] Empty or Invalid data structure")
            return
        # Log reference data if present
        for item in data["data_list"]:
            if item.get("rsp_type") == "A" and "references" in str(item):
                print(f"[DEBUG] Sending reference data: {json.dumps(data, ensure_ascii=False, indent=2)}")
        response = requests.post(url, json=data, timeout=10)
        
        if response.status_code != 200:
            print(f"[ERROR] Failed to send data: {response.status_code}, {response.text}")
        
        return response

    except Exception as e:
        print(f"[ERROR] send_data_to_server encountered an error: {str(e)}")
