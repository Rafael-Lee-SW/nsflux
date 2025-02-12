from vllm import LLM, SamplingParams

# 프롬프트 목록 정의
prompts = [
    "Mexico is famous for ",
    "The largest country in the world is "
]

# 샘플링 파라미터 설정
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# LLM 초기화
llm = LLM(model="facebook/opt-125m")

# 텍스트 생성
responses = llm.generate(prompts, sampling_params)

# 결과 출력
for response in responses:
    print(response.outputs[0].text)