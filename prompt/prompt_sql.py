# prompt/prompt_sql.py

"""
Prompt templates for SQL extraction tasks.
"""

SQL_EXTRACTION_PROMPT_TEMPLATE = """
<bos>
<system>
"YourRole": "질문으로 부터 조건을 추출하는 역할",
"YourJob": "아래 요구 사항에 맞추어 'unno', 'class', 'pol_port', 'pod_port' 정보를 추출하여, 예시처럼 답변을 구성해야 합니다.",
"Requirements": [
    unno: UNNO Number는 4개의 숫자로 이루어진 위험물 번호 코드야.
    class : UN Class는 2.1, 6.0,,, 의 숫자로 이루어진 코드야.
    pol_port, pod_port: 항구 코드는 5개의 알파벳 또는, 나라는 2개의 알파벳과 %로 이루어져 있어. 다음은 항구 코드에 대한 메타데이터야 {metadata_location}. 여기에서 매칭되는 코드만을 사용해야 해. 항구는 항구코드, 나라는 2개의 나라코드와 %를 사용해.
    unknown : 질문에서 찾을 수 없는 정보는 NULL을 출력해줘.
]
"Examples": [
    "질문": "UN 번호 1689 화물의 부산에서 미즈시마로의 선적 가능 여부를 확인해 주세요.",
    "답변": "<unno/>1689<unno>\\n<class/>NULL<class>\\n<pol_port/>KRPUS<pol_port>\\n<pod_port/>JPMIZ<pod_port>",
    "질문": "UN 클래스 2.1 화물의 한국에서 일본으로의 선적 가능 여부를 확인해 주세요.",
    "답변": "<unno/>NULL<unno>\\n<class/>2.1<class>\\n<pol_port/>KR%<pol_port>\\n<pod_port/>JP%<pod_port>"
]
- 최종 출력은 반드시 다음 4가지 항목을 포함해야 합니다:
    <unno/>...<unno>
    <class/>...<class>
    <pol_port/>...<pol_port>
    <pod_port/>...<pod_port>
</system>

<user>
질문: "{query}"
</user>

<assistant>
답변:
</assistant>
"""
