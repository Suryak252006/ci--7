
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)

# Generate 300 student records.
n = 300

data = pd.DataFrame({
    "attendance": rng.integers(40, 101, n),
    "internal_marks": rng.integers(20, 101, n),
    "assignment_marks": rng.integers(20, 101, n),
    "previous_score": rng.integers(20, 101, n),
})

# Generate labels using a weighted performance score.
score = (
    0.30 * data["attendance"]
    + 0.30 * data["internal_marks"]
    + 0.20 * data["assignment_marks"]
    + 0.20 * data["previous_score"]
)

data["result"] = (score >= 60).astype(int)
data.to_csv("student_results.csv", index=False)

features = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]

X = data[features]
y = data["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

with open("student_result_model.pkl", "wb") as file:
    pickle.dump(model, file)

metrics = {
    "accuracy": float(accuracy),
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model training completed.")
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, predictions))
print("Model and metrics saved.")
