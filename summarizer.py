# summarizer.py
# 이 파일은 LED-base-16384 모델을 사용하여 대화 텍스트를 여러 청크로 나눈 후 요약하는 기능을 제공합니다.
# 각 단계별로 입력 토큰 수, 청크 분할, 청크별 요약 결과, 최종 요약 결과 등을 로깅하여
# 문제가 발생하는 단계나 메커니즘을 추적할 수 있도록 합니다.

import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 로깅 설정 (필요에 따라 파일 핸들러 등 추가 가능)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# LED 모델 및 토크나이저 로드
led_model_name = "allenai/led-base-16384"
led_tokenizer = AutoTokenizer.from_pretrained(led_model_name)
led_tokenizer.pad_to_multiple_of = None  # 자동 패딩 비활성화

led_model = AutoModelForSeq2SeqLM.from_pretrained(led_model_name)
led_model.config.max_position_embeddings = 16384  # 긴 입력 지원

def summarize_conversation(conversation_text):
    logging.info("[Summarize] Called with conversation length: %d", len(conversation_text))
    
    # 청크당 최대 토큰 수 설정 (예: 2048 토큰)
    max_tokens_per_chunk = 2048

    # 전체 대화 텍스트를 토큰화하고 전체 토큰 수를 확인
    try:
        tokens = led_tokenizer.encode(conversation_text, add_special_tokens=True)
        total_tokens = len(tokens)
        logging.info("[Summarize] Total tokens: %d", total_tokens)
    except Exception as e:
        logging.error("[Summarize] Error during tokenization: %s", str(e))
        tokens = []
        total_tokens = 0

    # 청크 분할: 토큰 수가 max_tokens_per_chunk 이하면 그대로 사용,
    # 아니면 지정 길이만큼 청크로 나눕니다.
    chunks = []
    if total_tokens <= max_tokens_per_chunk:
        chunks = [conversation_text]
        logging.info("[Summarize] Input within max tokens; no chunk splitting required.")
    else:
        logging.info("[Summarize] Splitting conversation into chunks of max %d tokens.", max_tokens_per_chunk)
        for i in range(0, total_tokens, max_tokens_per_chunk):
            chunk_tokens = tokens[i:i+max_tokens_per_chunk]
            try:
                # decode를 통해 텍스트로 변환 (특수 토큰 제거)
                chunk_text = led_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            except Exception as e:
                logging.error("[Summarize] Error decoding tokens for chunk starting at index %d: %s", i, str(e))
                chunk_text = ""
            chunks.append(chunk_text)
        logging.info("[Summarize] Created %d chunks.", len(chunks))
    
    # 청크별 요약 수행: 각 청크에 대해 LED 모델로 요약 생성
    chunk_summaries = []
    for idx, chunk in enumerate(chunks):
        logging.info("[Summarize] Processing chunk %d with %d characters.", idx, len(chunk))
        try:
            # 입력 텐서 생성 (max_length 적용)
            inputs = led_tokenizer(chunk, return_tensors="pt", truncation=True, padding="longest", max_length=max_tokens_per_chunk)
            logging.info("[Summarize] Chunk %d input shape: %s", idx, inputs["input_ids"].shape)
            # 청크 요약 생성
            summary_ids = led_model.generate(
                inputs["input_ids"],
                max_length=150,
                min_length=30,
                no_repeat_ngram_size=3,
                temperature=0.7,
                num_beams=4,
                do_sample=True
            )
            summary_text = led_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            logging.info("[Summarize] Chunk %d summary: %s", idx, summary_text)
            chunk_summaries.append(summary_text)
        except Exception as e:
            logging.error("[Summarize] Exception summarizing chunk %d: %s", idx, str(e))
    
    if not chunk_summaries:
        logging.warning("[Summarize] No chunk summaries produced. Returning empty string.")
        return ""
    
    # 청크별 요약들을 결합
    combined_summary_text = " ".join(chunk_summaries)
    logging.info("[Summarize] Combined summary text length: %d", len(combined_summary_text))
    
    # 최종 요약: 결합된 요약 텍스트가 길 경우 다시 요약을 시도합니다.
    try:
        inputs = led_tokenizer(
            combined_summary_text, 
            return_tensors="pt", 
            truncation=True, 
            padding="longest", 
            max_length=max_tokens_per_chunk
        )
        logging.info("[Summarize] Combined summary input shape: %s", inputs["input_ids"].shape)
        final_summary_ids = led_model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=30,
            no_repeat_ngram_size=3,
            temperature=0.7,
            num_beams=4,
            do_sample=True
        )
        final_summary = led_tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)
        logging.info("[Summarize] Final combined summarization result: %s", final_summary)
    except Exception as e:
        logging.error("[Summarize] Exception during final summarization: %s", str(e))
        final_summary = combined_summary_text  # fallback으로 결합 요약 텍스트 사용

    # 모든 로깅 핸들러를 플러시합니다.
    for handler in logging.getLogger().handlers:
        handler.flush()
    
    return final_summary

# 테스트를 위한 main 함수
if __name__ == "__main__":
    sample_text = (
        "이것은 테스트 대화 내용입니다. 여러 문장이 포함되어 있으며, LED-base-16384 모델이 이를 어떻게 청크로 분할하고 요약하는지 확인합니다. "
        "예를 들어, 첫 번째 문장은 매우 길 수 있고, 두 번째 문장은 짧을 수 있습니다. "
        "이 텍스트는 실제 대화 데이터 대신 요약 기능을 테스트하기 위한 샘플입니다."
    )
    result = summarize_conversation(sample_text)
    print("Final Summary:", result)
