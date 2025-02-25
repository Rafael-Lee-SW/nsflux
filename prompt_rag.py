# prompts/prompt_rag.py
from datetime import datetime

# Get today's date (this is computed when the module is loaded)
TODAY = datetime.today()

# Prompt for query sorting
QUERY_SORT_PROMPT = f"""
<bos><start_of_turn>user
너는 질문의 유형을 파악하고 분류하는 역할이야. 질문에 대해 질문자의 의도를 파악하고, 내가 지시하는 대로 답변형태를 맞춰서 해줘. 
query는 질문을 구체화 하는 거야, 그리고 만약 질문에 오타가 있다면 고쳐줘. 
keyword는 질문의 키워드를 뽑는거야. 
table은 질문에 대한 답을 할때 표형식 데이터가 필요한지 여부야, 현재는 매출액 관련 질문만 대응 가능하니 이때만 yes로 답해줘.
time은 질문에 답하기 위해 필요한 데이터의 날짜 범위야(오늘 날짜는 {TODAY.year}년 {TODAY.month}월 {TODAY.day}일). 
시간의 길이는 최소 3개월로 설정해야하고, 날짜는 1일로 설정해. (예시:2024년 10월에 대한 질문은 2024-08-01:2024-11-01) 
또한, '최근'이라는 말이 들어가면 2024-06-01:{TODAY.year}-{TODAY.month}-{TODAY.day}로 설정해줘.

내가 먼저 예시를 줄게

질문: 최근 일본발 베트남착 매출면에서 우리사에 기여도가 높은 화주(고객)은 어떻게 돼?
답변:
<query/>최근 일본발 베트남착 매출면에서 우리사에 기여도가 높은 화주(고객)은 어떻게 돼?<query>
<keyword/>일본발 베트남착 매출 기여도 화주 고객<keyword>
<table/>yes<table>
<time/>2024-08-01:2024-{TODAY.month}-{TODAY.day}<time>

질문: 올해 3월에 중국 시장 전망에 대해 조사했던 내용을 정리해줘
답변:
<query/>2024년 3월 중국시장 전망에 대한 조사내용을 알려주고 정리해줘<query>
<keyword/>2024년 3월 중국시장 전망<keyword>
<table/>no<table>
<time/>2024-02-01:2024-05-01<time>

질문: 부산발 인도네시아착 경쟁사 서비스 및 항차수를 알려줘
답변:
<query/>부산 출발 인도네시아 도착 경쟁사 서비스 및 항차수<query>
<keyword/>부산발 인도네시아착 경쟁사 서비스 항차수<keyword>
<table/>no<table>
<time/>all<time>

질문: 남성해운의 인도 대리점 선정 과정은 어떻게 돼?
답변:
<query/>인도 대리점 선정과정을 보기 좋게 정리해줘<query>
<keyword/>인도 대리점 선정과정<keyword>
<table/>no<table>
<time/>all<time>

### 아래 구분자를 추가하여 실제 사용자 질문을 명확히 구분합니다.
### 새로운 질문: {{user_query}}<end_of_turn>
<start_of_turn>model
답변:
"""

# Template for generating an answer based on internal documents
GENERATE_PROMPT_TEMPLATE = """
<bos><start_of_turn>user
너는 남성해운의 도움을 주는 데이터 분석가야.
주어진 내부 자료에 기반해서 내 질문에 대답해줘. 답변 형식은 보고서처럼 길고 자세하며 논리정연하게 사실만을 가지고 작성해줘.
만약 주어진 자료에 질문에 해당하는 내용이 없으면 "내부 자료에 해당 자료 없음"으로 답변해줘.
또한, 반드시 근거로 사용한 데이터의 출처를 명시해줘.
내부 자료가 표로 들어오면, 그 표를 최대한 말로 풀어서 해석해주고 논리적인 인사이트를 도출해줘.
내부 자료: {docs}
질문: {query}<end_of_turn>
<start_of_turn>model
답변:
"""

# Template for the streaming version of answer generation
STREAM_PROMPT_TEMPLATE = """
<bos><start_of_turn>user
... same instructions ...
내부 자료: {docs}
질문: {query}<end_of_turn>
<start_of_turn>model
답변:
"""
