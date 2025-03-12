from flask import Flask, request
app = Flask(__name__)

@app.route("/testResponse", methods=["POST"])
def test_response():
    data = request.get_json()
    print("Received POST data:", data)
    return "OK", 200

if __name__ == "__main__":
    # 컨테이너 내부에서 포트 80 또는 6460번을 사용하도록 설정합니다.
    app.run(host="0.0.0.0", port=6460)
