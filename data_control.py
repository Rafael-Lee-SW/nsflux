import os
import json
import datetime
import torch
import numpy as np
from umap import UMAP
from flask import Blueprint, request, jsonify, render_template
from transformers import AutoModel, AutoTokenizer
from utils import vectorize_content  # Assumes you have defined vectorize_content in utils.py

# For PPTX extraction
from pptx import Presentation

# For PDF extraction
import PyPDF2

# For visualization we use Plotly
import plotly.express as px
import pandas as pd

DATA_PATH = "data/1104_NS_DB_old.json"

# 임베딩 모델 로드 (필요한 경우 캐싱 고려)
embedding_model = AutoModel.from_pretrained("BM-K/KoSimCSE-roberta-multitask")
embedding_tokenizer = AutoTokenizer.from_pretrained("BM-K/KoSimCSE-roberta-multitask")
embedding_model.eval()

data_control_bp = Blueprint("data_manager", __name__, template_folder="templates")

# --- Helper functions for new file types ---

def extract_text_from_pptx(file_path):
    prs = Presentation(file_path)
    texts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                texts.append(shape.text)
    return "\n".join(texts)

def extract_text_from_pdf(file_path):
    pdf_text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text + "\n"
    return pdf_text

@data_control_bp.route("/manager")
def data_control_page():
    return render_template("data_manager.html")


@data_control_bp.route("/upload", methods=["POST"])
def data_upload():
    if "dataFile" not in request.files:
        return jsonify({"message": "파일이 업로드되지 않았습니다."}), 400
    file = request.files["dataFile"]
    if file.filename == "":
        return jsonify({"message": "파일 이름이 없습니다."}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    try:
        # TXT 파일 처리
        if ext == ".txt":
            content = file.read().decode("utf-8")
            vector = vectorize_content(content)
            
            # 벡터 검증 추가
            if not vector or not isinstance(vector, list):
                return jsonify({"message": "벡터화 실패: 유효하지 않은 벡터 생성"}), 500
            
            # 벡터 차원 확인 (임베딩 모델에 맞게 조정)
            expected_dim = 768
            if len(vector) != expected_dim:
                print(f"경고: 벡터 차원 불일치. 예상: {expected_dim}, 실제: {len(vector)}")
                if len(vector) < expected_dim:
                    vector.extend([0.0] * (expected_dim - len(vector)))
                else:
                    vector = vector[:expected_dim]
            
            new_entry = {
                "file_name": file.filename,
                "chunks": [
                    {
                        "chunk_id": 1,
                        "title": os.path.splitext(file.filename)[0],
                        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "type": "text",
                        "text": content,
                        "text_short": content[:200],
                        "vector": vector,
                        "text_vis": content,
                    }
                ],
            }
            
        # JSON 파일 처리
        elif ext == ".json":
            new_entry = json.load(file)
            if not isinstance(new_entry, list):
                new_entry = [new_entry]
            
            # JSON 파일의 각 청크에 대해 벡터 검증
            for entry in new_entry:
                for chunk in entry.get("chunks", []):
                    if not chunk.get("vector"):
                        text = chunk.get("text", "")
                        if text:
                            chunk["vector"] = vectorize_content(text)
                        else:
                            # 텍스트가 없으면 기본 벡터 사용
                            chunk["vector"] = [0.0] * 768
                            print(f"경고: 텍스트 없는 청크에 기본 벡터 사용")
                    else:
                        # 기존 벡터 검증
                        vector = chunk["vector"]
                        expected_dim = 768
                        
                        if not isinstance(vector, list):
                            print(f"경고: 벡터가 리스트가 아님, 텍스트 재벡터화")
                            text = chunk.get("text", "")
                            if text:
                                chunk["vector"] = vectorize_content(text)
                            else:
                                chunk["vector"] = [0.0] * expected_dim
                        elif len(vector) != expected_dim:
                            print(f"경고: 벡터 차원 불일치. 예상: {expected_dim}, 실제: {len(vector)}")
                            if len(vector) < expected_dim:
                                chunk["vector"] = vector + [0.0] * (expected_dim - len(vector))
                            else:
                                chunk["vector"] = vector[:expected_dim]
                    
        # 다른 파일 형식 처리는 동일한 방식으로 벡터 검증 로직 추가
        # ...

        # 기존 데이터 로드 및 새 데이터 추가
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = []
            
        if isinstance(new_entry, list):
            existing_data.extend(new_entry)
        else:
            existing_data.append(new_entry)
            
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
        return jsonify({"message": "파일 업로드 및 벡터화가 완료되었습니다."})
    except Exception as e:
        return jsonify({"message": f"업로드 실패: {str(e)}"}), 500


@data_control_bp.route("/list", methods=["GET"])
def data_list():
    try:
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                data_entries = json.load(f)
        else:
            data_entries = []
        summary = []
        for entry in data_entries:
            if "chunks" in entry and entry["chunks"]:
                chunk = entry["chunks"][0]
                summary.append({
                    "file_name": entry.get("file_name", ""),
                    "title": chunk.get("title", ""),
                    "date": chunk.get("date", ""),
                })
            else:
                summary.append({
                    "file_name": entry.get("file_name", ""),
                    "title": "",
                    "date": ""
                })
        return jsonify(summary)
    except Exception as e:
        return jsonify({"message": f"데이터 목록 불러오기 실패: {str(e)}"}), 500

@data_control_bp.route("/delete", methods=["POST"])
def data_delete():
    try:
        req = request.get_json()
        index = req.get("index")
        if index is None:
            return jsonify({"message": "삭제할 인덱스가 제공되지 않았습니다."}), 400
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                data_entries = json.load(f)
        else:
            return jsonify({"message": "데이터 파일이 존재하지 않습니다."}), 404
        if not (0 <= index < len(data_entries)):
            return jsonify({"message": "유효하지 않은 인덱스입니다."}), 400
        del data_entries[index]
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(data_entries, f, ensure_ascii=False, indent=2)
        return jsonify({"message": "데이터 삭제가 완료되었습니다."})
    except Exception as e:
        return jsonify({"message": f"데이터 삭제 실패: {str(e)}"}), 500

# --- Search and highlight endpoint ---
@data_control_bp.route("/search", methods=["GET"])
def search_data():
    query = request.args.get("q", "").lower()
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data_entries = json.load(f)
    else:
        data_entries = []
    results = []
    for idx, entry in enumerate(data_entries):
        file_name = entry.get("file_name", "").lower()
        title = ""
        if "chunks" in entry and entry["chunks"]:
            title = entry["chunks"][0].get("title", "").lower()
        if query in file_name or query in title:
            results.append({
                "index": idx,
                "file_name": entry.get("file_name", ""),
                "title": title,
            })
    return jsonify(results)


@data_control_bp.route("/umap_visualization")
def umap_visualization_page():
    return render_template("umap_visualization.html")

@data_control_bp.route("/api/umap_data", methods=["GET"])
def get_umap_data():
    try:
        # Load vector data
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                data_entries = json.load(f)
        else:
            return jsonify({"error": "데이터 파일을 찾을 수 없습니다."}), 404
        
        # Extract vectors and metadata
        vectors = []
        metadata = []
        
        for entry in data_entries:
            for chunk in entry.get("chunks", []):
                if "vector" in chunk and chunk["vector"]:
                    vectors.append(chunk["vector"])
                    metadata.append({
                        "id": f"{entry.get('file_name', 'unknown')}_{chunk.get('chunk_id', '0')}",
                        "file_name": entry.get("file_name", "Unknown"),
                        "title": chunk.get("title", ""),
                        "date": chunk.get("date", ""),
                        "text_short": chunk.get("text_short", "")[:100] + "..."
                    })
        
        if not vectors:
            return jsonify({"error": "벡터 데이터가 없습니다."}), 404
        
        # Convert to numpy array and squeeze extra dimensions
        vectors_array = np.array(vectors)
        vectors_array = np.squeeze(vectors_array)
        
        # Apply UMAP for dimensionality reduction
        from umap import UMAP
        n_neighbors = min(15, len(vectors) - 1)  # Adjust based on data size
        umap_model = UMAP(n_components=2, 
                            n_neighbors=n_neighbors, 
                            min_dist=0.1, 
                            metric='cosine', 
                            random_state=42)
        embedding = umap_model.fit_transform(vectors_array)
        
        # Prepare response data
        nodes = []
        for i, (x, y) in enumerate(embedding):
            nodes.append({
                "id": metadata[i]["id"],
                "x": float(x),
                "y": float(y),
                "file_name": metadata[i]["file_name"],
                "title": metadata[i]["title"],
                "date": metadata[i]["date"],
                "text_short": metadata[i]["text_short"]
            })
        
        # Calculate edges (connections between similar vectors)
        edges = []
        if len(vectors) > 1:
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(vectors_array)
            threshold = 0.7
            for i in range(len(vectors)):
                for j in range(i+1, len(vectors)):
                    if similarity[i, j] > threshold:
                        edges.append({
                            "source": metadata[i]["id"],
                            "target": metadata[j]["id"],
                            "value": float(similarity[i, j])
                        })
        
        return jsonify({
            "nodes": nodes,
            "edges": edges
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"UMAP 시각화 처리 중 오류 발생: {str(e)}"}), 500