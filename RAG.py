import torch
import re
import numpy as np
import rank_bm25
import torch
import numpy as np
import random
from datetime import datetime, timedelta
from sql import generate_sql

global beep
beep = '-------------------------------------------------------------------------------------------------------------------------------------------------------------------------'

def execute_rag(QU,KE,TA,TI, **kwargs):
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    embed_model = kwargs.get("embed_model")
    embed_tokenizer = kwargs.get("embed_tokenizer")
    data = kwargs.get("data")
    config = kwargs.get("config")

    if TA == "yes": # Table 이 필요하면
        # SQL
        SQL_results = generate_sql(QU, model, tokenizer, config)
        # answer = generate(SQL_results,query, model, tokenizer, config)
        return SQL_results, None

    else:
        # RAG
        data = sort_by_time(TI, data)
        docs, docs_list = retrieve(KE, data, config.N, embed_model, embed_tokenizer)
        # answer = generate(docs, QU, model, tokenizer, config)
        return docs, docs_list

def generate_answer(query, docs, **kwargs):
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    config = kwargs.get("config")

    answer = generate(docs, query, model, tokenizer, config)
    return answer
    

def query_sort(query, **kwargs):
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    config = kwargs.get("config")

    PROMPT =\
f'''\
<bos><start_of_turn>user
너는 질문의 유형을 파악하고 분류하는 역할이야. 질문에 대해 질문자의 의도를 파악하고, 내가 지시하는 대로 답변형태를 맞춰서 해줘. \
query는 질문을 구체화 하는 거야, 그리고 만약 질문에 오타가 있다면 고쳐줘. \
keyword는 질문의 키워드를 뽑는거야. \
table은 질문에 대한 답을 할때 표형식 데이터가 필요한지 여부야, 현재는 매출액 관련 질문만 대응 가능하니 이때만 yes로 답해줘.\
time은 질문에 답하기 위해 필요한 데이터의 날짜 범위야(오늘 날짜는{datetime.today().year}년 {datetime.today().month}월 {datetime.today().day}일). 시간의 길이는 최소 3개월로 설정해야하고, 날짜는 1일로 설정해. (예시:2024년 10월에 대한 질문은 2024-08-01:2024-11-01) \
또한, '최근'이라는 말이 들어가면 2024-08-01:{datetime.today().year}-{datetime.today().month}-{datetime.today().day}로 설정해줘.\

내가 먼저 예시를 줄게

질문: 최근 일본발 베트남착 매출면에서 우리사에 기여도가 높은 화주(고객)은 어떻게 돼?
답변:
<query/>최근 일본발 베트남착 매출면에서 우리사에 기여도가 높은 화주(고객)은 어떻게 돼?<query>
<keyword/>일본발 베트남착 매출 기여도 화주 고객<keyword>
<table/>yes<table>
<time/>2024-08-01:2024-{datetime.today().month}-{datetime.today().day}<time>

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

위에서 알려준대로 다음 질문에 대해서 답변을 생성해줘
질문: {query}<end_of_turn>
<start_of_turn>model
답변: \
'''
    # Get Answer
    input_ids = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    input_length = input_ids['input_ids'].shape[1]
    outputs = model.generate(
        **input_ids,
        max_new_tokens=config.model.max_new_tokens,     # 생성할 최대 토큰 수
        do_sample = config.model.do_sample,
        temperature = config.model.temperature,           # 텍스트 다양성 조정
        top_k = config.model.top_k,                       # top-k 샘플링
        top_p = config.model.top_p,                       # top-p(누적 확률) 샘플링
        repetition_penalty = config.model.repetition_penalty,       # 반복 패턴 억제
        eos_token_id=tokenizer.eos_token_id,         # 조기 종료 토큰 (EOS 토큰이 있을 경우 종료)
        pad_token_id=tokenizer.eos_token_id          # 패딩 시 EOS 토큰 사용
        )

    answer = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    # Regular expression to extract content between <query/> and <query>
    query_pattern = r'<query/>(.*?)<query>'
    keyword_pattern = r'<keyword/>(.*?)<keyword>'
    table_pattern = r'<table/>(.*?)<table>'
    time_pattern = r'<time/>(.*?)<time>'

    QU = re.search(query_pattern, answer, re.DOTALL).group(1)
    KE = re.search(keyword_pattern, answer, re.DOTALL).group(1)
    TA = re.search(table_pattern, answer, re.DOTALL).group(1)
    TI = re.search(time_pattern, answer, re.DOTALL).group(1)
    
    # 시간이 all 이면 전체에서 검색.
    if TI == "all":
        TI = "1900-01-01:2099-01-01"
    print(beep)
    print(f'사용자 질문:{query}')
    print(beep)
    print(f'구체화 질문: {QU}, 키워드 : {KE}, 테이블 필요 유무: {TA}, 시간: {TI}')
    print(beep)

    return QU, KE, TA, TI

def sort_by_time(time_bound, data):
    date_format = "%Y-%m-%d"
    target_date_start = datetime.strptime(time_bound.split(":")[0], date_format)
    target_date_end   = datetime.strptime(time_bound.split(":")[1], date_format)

    matching_indices = [
                        i for i, date in enumerate(data['times']) 
                        if (not isinstance(date, str)) and (target_date_start < date < target_date_end)
                        ]


    data['file_names'], data['titles'], data['times'], data['vectors'], data['texts'], data['texts_short'], data['texts_vis'] = ([lst[i] for i in matching_indices] for lst in (data['file_names'], data['titles'], data['times'], data['vectors'], data['texts'], data['texts_short'], data['texts_vis']))
    return data

def retrieve(query, data, N, embed_model, embed_tokenizer):
    # Similarity Score
    sim_score = cal_sim_score(query, data['vectors'], embed_model, embed_tokenizer)

    # BM25 Score
    bm25_score = cal_bm25_score(query, data['texts'], embed_tokenizer)

    # Scaling Scores
    scaled_sim_score = min_max_scaling(sim_score)
    scaled_bm25_score = min_max_scaling(bm25_score)

    # Total Score
    score = (scaled_sim_score + scaled_bm25_score) / 2
    top_k = score[:,0,0].argsort()[-N:][::-1]

    ## documents string 버전, dictionary 버전 둘 다 필요.
    documents = ""
    documents_list = []
    for i,index in enumerate(top_k):
        documents += f"{i+1}번째 검색자료 (출처:{data['file_names'][index]}) :\n{data['texts_short'][index-1]}{data['texts_short'][index]}{data['texts_short'][index+1]}\n"
        documents_list.append(data['texts_vis'][index])
        print(f"\n{i+1}번째 검색자료 (출처:{data['file_names'][index]}) :\n{data['texts_short'][index-1]}{data['texts_short'][index]}{data['texts_short'][index+1]}")
        print('\n'+beep)
    
    return documents, documents_list

def cal_sim_score(query, chunks, embed_model, embed_tokenizer):
    query_V = embed(query, embed_model, embed_tokenizer)
    if len(query_V.shape) == 1: query_V = query_V.unsqueeze(0)

    score = []
    for chunk in chunks:
        if len(chunk.shape) == 1: chunk = chunk.unsqueeze(0)

        query_norm = query_V / query_V.norm(dim=1)[:, None]
        chunk_norm = chunk / chunk.norm(dim=1)[:, None]
        tmp = torch.mm(query_norm, chunk_norm.transpose(0, 1)) * 100
        score.append(tmp.detach())

    return np.array(score)

def cal_bm25_score(query, indexes, embed_tokenizer):
    tokenized_corpus = [embed_tokenizer(text, return_token_type_ids=False, return_attention_mask=False, return_offsets_mapping=False) for text in indexes]
    tokenized_corpus = [embed_tokenizer.convert_ids_to_tokens(corpus['input_ids']) for corpus in tokenized_corpus]

    bm25 = rank_bm25.BM25Okapi(tokenized_corpus)

    tokenized_query = embed_tokenizer(query)
    tokenized_query = embed_tokenizer.convert_ids_to_tokens(tokenized_query['input_ids'])
    bm25_score = bm25.get_scores(tokenized_query)

    return np.array(bm25_score)

def embed(query, embed_model, embed_tokenizer):
    inputs = embed_tokenizer(query, padding=True, truncation=True, return_tensors="pt")
    embeddings, _ = embed_model(**inputs, return_dict=False)
    return embeddings[0][0]

def min_max_scaling(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def generate(docs, query, model, tokenizer, config):
    PROMPT =\
f'''
<bos><start_of_turn>user
주어진 내부 자료에 기반해서 내 질문에 대답해줘. 답변 형식은 보고서처럼 길고 자세하고 논리정연하게 사실만을 가지고 작성해줘.  만약 주어진 자료에 질문에 해당하는 내용이 없으면 "내부 자료에 해당 자료 없음"으로 답변해줘. 또한, 반드시 근거로 사용한 데이터의 출처를 명시해줘.
내부 자료: {docs}
질문: {query}<end_of_turn>
<start_of_turn>model
답변: \
'''
    input_ids = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    input_length = input_ids['input_ids'].shape[1]
    print(f"전체 입력 토큰 수:{input_length}")
    outputs = model.generate(
        **input_ids,
        max_new_tokens=config.model.max_new_tokens,     # 생성할 최대 토큰 수
        do_sample = config.model.do_sample,
        temperature = config.model.temperature,           # 텍스트 다양성 조정
        top_k = config.model.top_k,                       # top-k 샘플링
        top_p = config.model.top_p,                       # top-p(누적 확률) 샘플링
        repetition_penalty = config.model.repetition_penalty,       # 반복 패턴 억제
        eos_token_id=tokenizer.eos_token_id,         # 조기 종료 토큰 (EOS 토큰이 있을 경우 종료)
        pad_token_id=tokenizer.eos_token_id          # 패딩 시 EOS 토큰 사용
        )

    answer = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return answer

if __name__ == '__main__':
    status = True

    while status == True:
        query = input("질문 : ")
        QU,TA,TI = query_sort(query)

        if TA == "yes": # Table 이 필요하면
            print('\n'+beep)
            SQL_results = generate_sql(QU)
            answer = generate(SQL_results,query)
            print(answer)
            print('\n'+beep)
            print('\n'+beep)
            print('\n'+beep)

        else:
            file_names, titles, times, vectors, texts, texts_short = sort_by_time(TI, file_names, titles, times, vectors, texts, texts_short)
            print('\n'+beep)
            docs = retrieve(QU, vectors, texts, texts_short, file_names, N)
            print('\n'+beep)
            answer = generate(docs,query)
            print(answer)
            print('\n'+beep)
            print('\n'+beep)
            print('\n'+beep)
        