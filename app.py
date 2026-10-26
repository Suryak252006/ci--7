
import pickle

from flask import Flask, jsonify, request

app = Flask(__name__)

with open("student_result_model.pkl", "rb") as file:
    model = pickle.load(file)

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "student-result-prediction",
        "status": "ok"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object"}), 400

    try:
        values = [float(data[name]) for name in FEATURES]
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Invalid or missing features"}), 400

    if not all(0 <= value <= 100 for value in values):
        return jsonify({"error": "Feature values must be between 0 and 100"}), 400

    prediction = int(model.predict([values])[0])

    return jsonify({
        "prediction": "PASS" if prediction == 1 else "FAIL",
        "prediction_code": prediction
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
