
import json
import sys

MIN_ACCURACY = 0.85

with open("metrics.json") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print(f"Model accuracy: {accuracy:.4f}")
print(f"Required accuracy: {MIN_ACCURACY:.2f}")

if accuracy < MIN_ACCURACY:
    print("QUALITY GATE FAILED")
    sys.exit(1)

print("QUALITY GATE PASSED")
