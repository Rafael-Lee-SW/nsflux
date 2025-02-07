# ray_inference.py
import ray
import os
from utils import load_model, load_data, random_seed
from RAG import query_sort, execute_rag, generate_answer
from utils import process_format_to_response, error_format
import torch

# 선택 옵션: vLLM 사용 여부 (config 또는 환경변수로 결정할 수 있음)
USE_VLLM = os.getenv("USE_VLLM", "False").lower() == "true"

if USE_VLLM:
    from vllm import LLM, SamplingParams

    # vLLM 초기화 예시 (모델명, 토크나이저 등은 상황에 맞게 설정)
    # 주의: vLLM은 huggingface와 API가 다르므로 기존 코드와 약간의 수정이 필요합니다.
    # 아래는 간단한 예시입니다.

    class VLLMEngine:
        def __init__(self, model_name, sampling_params):
            self.llm = LLM(
                model=model_name
            )  # vLLM 엔진 생성 (내부적으로 모델과 토크나이저 로딩)
            self.sampling_params = sampling_params

        def generate(self, prompt, max_new_tokens):
            # vLLM API 호출 예시 (실제 사용 시 vLLM 문서를 참고하세요)
            outputs = self.llm.generate(
                [prompt], self.sampling_params, max_new_tokens=max_new_tokens
            )
            # outputs는 리스트 형태로 반환되며, 각 출력은 객체로 text 속성을 가지고 있음
            return outputs[0].text

else:
    VLLMEngine = None  # 사용하지 않을 경우 None

@ray.remote
class RAGActor:
    def __init__(self, config):
        # 초기화: 랜덤 시드 설정 및 모델/데이터 로드
        random_seed(config.seed)
        self.config = config

        # 기존 Huggingface 모델 로딩 (utils.py 내 load_model)
        self.model, self.tokenizer, self.embed_model, self.embed_tokenizer = load_model(
            config
        )
        self.data = load_data(config.data_path)

        # 만약 vLLM을 사용한다면 vLLM 엔진을 초기화
        if USE_VLLM:
            sampling_params = SamplingParams(
                temperature=config.model.temperature,
                top_k=config.model.top_k,
                top_p=config.model.top_p,
                # 기타 vLLM 관련 파라미터 설정
            )
            self.vllm_engine = VLLMEngine(config.model_id, sampling_params)
        else:
            self.vllm_engine = None

    def inference(self, user_input):
        """
        주어진 사용자 입력에 대해 RAG 시스템(질문 분류, 검색, 답변 생성)을 실행하고,
        결과를 반환합니다.
        만약 vLLM을 사용하도록 설정되어 있다면, generate_answer 대신 vLLM 엔진을 사용합니다.
        """
        # Step 1. 질문 분류 및 구체화: query_sort
        QU, KE, TA, TI = query_sort(
            user_input, model=self.model, tokenizer=self.tokenizer, config=self.config
        )

        # Step 2. RAG 실행: SQL 또는 단순 Retrieval 분기 처리
        if TA == "yes":
            try:
                docs, docs_list = execute_rag(
                    QU,
                    KE,
                    TA,
                    TI,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    embed_model=self.embed_model,
                    embed_tokenizer=self.embed_tokenizer,
                    data=self.data,
                    config=self.config,
                )
                retrieval, chart = (
                    docs,
                    docs_list,
                )  # process_to_format 내부 처리 (생략 가능)
                # 답변 생성
                if self.vllm_engine:
                    # vLLM 사용: prompt 생성 후 vLLM 엔진 호출
                    prompt = (
                        QU + "\n" + retrieval
                    )  # 예시 prompt 구성 (원하는 형식으로 수정)
                    output = self.vllm_engine.generate(
                        prompt, max_new_tokens=self.config.model.max_new_tokens
                    )
                else:
                    output = generate_answer(
                        QU,
                        docs,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        config=self.config,
                    )
                answer = process_format_to_response(retrieval, output)
                return answer
            except Exception as e:
                return error_format(
                    f"내부 Excel 에 해당 자료가 없습니다. 오류: {str(e)}", 551
                )
        else:
            try:
                docs, docs_list = execute_rag(
                    QU,
                    KE,
                    TA,
                    TI,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    embed_model=self.embed_model,
                    embed_tokenizer=self.embed_tokenizer,
                    data=self.data,
                    config=self.config,
                )
                retrieval = docs  # Retrieval 방식 처리
                if self.vllm_engine:
                    prompt = QU + "\n" + retrieval
                    output = self.vllm_engine.generate(
                        prompt, max_new_tokens=self.config.model.max_new_tokens
                    )
                else:
                    output = generate_answer(
                        QU,
                        docs,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        config=self.config,
                    )
                answer = process_format_to_response(retrieval, output)
                return answer
            except Exception as e:
                return error_format(
                    f"내부 PPT 에 해당 자료가 없습니다. 오류: {str(e)}", 552
                )
