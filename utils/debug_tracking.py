import os
import torch
import psutil
import logging

def log_system_info(label=""):
    """현재 프로세스와 GPU 메모리 사용량을 로깅합니다."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
    logging.info(f"{label} - Process Memory: RSS={mem_info.rss/1024**2:.2f} MB, VMS={mem_info.vms/1024**2:.2f} MB")
    logging.info(f"{label} - GPU Memory: allocated={allocated/1e6:.2f} MB, reserved={reserved/1e6:.2f} MB")

def log_batch_info(batch):
    """현재 배치의 크기와 각 요청의 간단한 토큰 길이(공백 기준)를 로깅합니다."""
    batch_size = len(batch)
    token_counts = []
    for item in batch:
        # item은 (http_query, future, sse_queue) 튜플
        http_query = item[0]
        # http_query가 dict라면 qry_contents를 가져옵니다.
        query = http_query.get("qry_contents", "") if isinstance(http_query, dict) else ""
        tokens = query.split()
        token_counts.append(len(tokens))
    logging.info(f"[Batch Tracking] Batch size: {batch_size}, Token counts: {token_counts}")
