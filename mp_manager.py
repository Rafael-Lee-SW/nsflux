# mp_manager.py
import ray
import logging
from RAG import query_sort, execute_rag, generate_answer
from utils import load_model, load_data, random_seed
import json

# Optionally: from your config loader
import yaml
from box import Box

logging.basicConfig(level=logging.INFO)

# 1) Initialize Ray
#    If you run on CPU only, you can limit the number of workers, etc.
ray.init(
    # For local CPU dev: 
    # num_cpus=4,  # or however many you want
    # You can omit this if you want auto-detection
)

# 2) Load your global model + data ONCE in the driver process
#    (You can also choose to load inside the worker if you prefer.)
with open('./config.yaml', 'r', encoding='utf-8') as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    config = Box(config_yaml)
random_seed(config.seed)

model, tokenizer, embed_model, embed_tokenizer = load_model(config)
data = load_data(config.data_path)

# 3) Prepare a "keyword arguments" dict if you want to pass them around easily
global_kwargs = {
    "model": model,
    "tokenizer": tokenizer,
    "embed_model": embed_model,
    "embed_tokenizer": embed_tokenizer,
    "data": data,
    "config": config,
}


# 4) Make a Ray task or Actor that wraps your RAG pipeline

@ray.remote
def rag_pipeline(qry_contents: str) -> dict:
    """
    This function runs your query pipeline (query_sort -> execute_rag -> generate_answer)
    inside a Ray remote function, returning the final JSON/dict result.
    """
    # Re-load data if you want fresh data each time (or skip if not needed)
    # data = load_data(config.data_path)
    # local_kwargs = dict(global_kwargs)
    # local_kwargs["data"] = data

    # We can pass in the "global_kwargs" in many ways. For simplicity:
    local_kwargs = global_kwargs

    # 1) query_sort
    QU, KE, TA, TI = query_sort(qry_contents, **local_kwargs)

    # 2) execute_rag
    docs, docs_list = execute_rag(QU, KE, TA, TI, **local_kwargs)

    # 3) generate_answer
    answer_text = generate_answer(QU, docs, **local_kwargs)

    # Return all or part of the data
    final_response = {
        "QU": QU,
        "KE": KE,
        "TA": TA,
        "TI": TI,
        "docs_list": docs_list,
        "answer": answer_text
    }
    return final_response


# Optional: A test main
if __name__ == "__main__":
    # Example usage in a local test (not using Flask):
    query_str = "남성해운 디지털 전략 알려줘"
    future = rag_pipeline.remote(query_str)
    result = ray.get(future)
    print("Ray pipeline result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
