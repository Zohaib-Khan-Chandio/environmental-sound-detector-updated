"""
Environmental Sound Detector - web backend.

Run:   python app.py        then open  http://localhost:5000
Needs these files in the SAME folder:
    app.py, index.html, features_v2.py, model.pkl, classes.json, config.json
(model.pkl / classes.json / config.json are created by Export_For_Website.ipynb)

The page sends the browser-decoded audio (mono, 22050 Hz, float32, first 4 s) to /api/predict.
This server turns it into the SAME 316 features used in training (features_v2.py),
runs the trained model, and returns the top predictions.
"""
import io
import json
import os
import sys

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features_v2  # noqa: E402  (the exact feature code used for training)

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(HERE, "model.pkl"))
CLASSES_PATH = os.path.join(HERE, "classes.json")
CONFIG_PATH = os.path.join(HERE, "config.json")

# ---------------------------------------------------------------- load model
for p in (MODEL_PATH, CLASSES_PATH, CONFIG_PATH):
    if not os.path.exists(p):
        sys.exit(f"Missing file: {p}\nRun Export_For_Website.ipynb in Colab first and copy the 'website' folder here.")

model = joblib.load(MODEL_PATH)
CLASSES = json.load(open(CLASSES_PATH))
CONFIG = json.load(open(CONFIG_PATH))
THRESHOLD = float(CONFIG.get("threshold", 0.55))
MODEL_NAME = CONFIG.get("model_name", type(model).__name__)

if not hasattr(model, "predict_proba"):
    sys.exit("This model cannot give probabilities (an SVC trained with probability=False). "
             "Use the model.pkl made by Export_For_Website.ipynb.")
if type(model).__name__ == "SVC":
    sys.exit("model.pkl is a bare SVC without its scaler. Use the model.pkl made by Export_For_Website.ipynb "
             "(it is a Pipeline: StandardScaler + SVC).")

N_FEATURES = len(features_v2.FEATURE_COLUMNS)
if hasattr(model, "n_features_in_") and model.n_features_in_ != N_FEATURES:
    sys.exit(f"Model expects {model.n_features_in_} features but features_v2.py makes {N_FEATURES}.")

# model.classes_ holds the class ids in the order of predict_proba columns
CLASS_IDS = [int(c) for c in getattr(model, "classes_", range(len(CLASSES)))]
LABELS = [CLASSES[i] for i in CLASS_IDS]

# warm up librosa/numba so the first user request is not slow
try:
    features_v2.extract_features_v2((np.random.randn(features_v2.SR * 2) * 0.05).astype("float32"))
except Exception as e:  # pragma: no cover
    print("warm-up warning:", e)

# ---------------------------------------------------------------------- app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024


@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.get("/api/info")
def info():
    import sklearn
    return jsonify({
        "model": MODEL_NAME,
        "classes": LABELS,
        "threshold": THRESHOLD,
        "sample_rate": features_v2.SR,
        "n_features": N_FEATURES,
        "sklearn": sklearn.__version__,
        "trained_with_sklearn": CONFIG.get("sklearn_version"),
        "fold10_accuracy": CONFIG.get("fold10_accuracy"),
        "fold10_macro_f1": CONFIG.get("fold10_macro_f1"),
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def _read_audio():
    """Returns mono float32 audio at features_v2.SR."""
    if request.mimetype and request.mimetype.startswith("multipart"):
        # convenience path (curl / testing): uploaded audio file, decoded with librosa
        import librosa
        f = request.files.get("file")
        if f is None:
            raise ValueError("send the audio as form field 'file'")
        y, _ = librosa.load(io.BytesIO(f.read()), sr=features_v2.SR, mono=True)
        return y.astype("float32")

    raw = request.get_data()
    if len(raw) < 4 or len(raw) % 4:
        raise ValueError("body must be raw float32 samples")
    y = np.frombuffer(raw, dtype="<f4").astype("float32")
    sr = int(request.args.get("sr", features_v2.SR))
    if sr != features_v2.SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=features_v2.SR)
    return y


@app.post("/api/predict")
def predict():
    try:
        y = _read_audio()
    except Exception as e:
        return jsonify({"error": f"Could not read audio: {e}"}), 400

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if len(y) < 200:
        return jsonify({"error": "Audio is too short."}), 400

    feats = features_v2.extract_features_v2(y)
    if feats is None:
        return jsonify({"error": "Feature extraction failed for this audio."}), 422

    proba = model.predict_proba(feats.reshape(1, -1))[0]
    order = np.argsort(proba)[::-1]
    top = [{"label": LABELS[i], "prob": float(proba[i])} for i in order[:3]]
    best = top[0]
    return jsonify({
        "label": best["label"],
        "confidence": best["prob"],
        "threshold": THRESHOLD,
        "accepted": best["prob"] >= THRESHOLD,
        "top3": top,
        "probs": {LABELS[i]: float(proba[i]) for i in range(len(LABELS))},
        "seconds_analysed": round(min(len(y), features_v2.MAX_LEN) / features_v2.SR, 2),
        "model": MODEL_NAME,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nModel: {MODEL_NAME} | threshold {THRESHOLD:.2f}\nOpen  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
