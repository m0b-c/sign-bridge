import os, json, re, uuid
from collections import Counter
from flask import Flask, jsonify, render_template, request, send_from_directory
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SIGNBRIDGE_DATA_DIR", "/tmp/signbridge_data" if os.environ.get("VERCEL") else os.path.join(BASE, "data"))
SAMPLES_FILE = os.path.join(DATA_DIR, "sequence_samples.json")
MODEL_FILE = os.path.join(DATA_DIR, "sequence_model.joblib")
CLIPS_DIR = os.path.join(DATA_DIR, "sign_clips")
os.makedirs(CLIPS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
model = None
model_info = {"ready": False, "classes": [], "accuracy": None, "samples": 0}

# Each frame is 2 hands * 21 landmarks * xyz + 2 presence flags = 128 values.
# Each captured example is a fixed-length sequence of 24 frames.
SEQUENCE_FRAMES = 24
FEATURES_PER_FRAME = 128
ALLOWED_LABEL = re.compile(r"^[A-Z0-9][A-Z0-9 _'-]{0,29}$")

def load_rows():
    if not os.path.exists(SAMPLES_FILE):
        return []
    try:
        with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def load_model_if_present():
    global model, model_info
    try:
        import joblib
        if os.path.exists(MODEL_FILE):
            model = joblib.load(MODEL_FILE)
            model_info.update(ready=True, classes=list(model.classes_))
    except Exception:
        model = None

@app.get("/")
def home():
    return render_template("index.html", sequence_frames=SEQUENCE_FRAMES)

@app.get("/manifest.webmanifest")
def manifest():
    return send_from_directory(BASE, "manifest.webmanifest", mimetype="application/manifest+json")

@app.get("/sw.js")
def service_worker():
    return send_from_directory(os.path.join(BASE, "public"), "sw.js", mimetype="application/javascript")

@app.get("/api/status")
def status():
    rows = load_rows()
    counts = Counter(r["label"] for r in rows)
    return jsonify(
        samples=len(rows),
        classes=dict(counts),
        model_ready=model is not None,
        model_classes=list(model.classes_) if model is not None else [],
        sequence_frames=SEQUENCE_FRAMES,
        clips=len([f for f in os.listdir(CLIPS_DIR) if f.lower().endswith((".mp4", ".webm", ".mov"))])
    )

@app.post("/api/samples")
def add_sample():
    body = request.get_json(silent=True) or {}
    label = str(body.get("label", "")).strip().upper()
    sequence = body.get("sequence")
    if not ALLOWED_LABEL.fullmatch(label):
        return jsonify(ok=False, error="Use a short label: A-Z, 0-9, or a word such as HELLO."), 400
    if not isinstance(sequence, list) or len(sequence) != SEQUENCE_FRAMES:
        return jsonify(ok=False, error=f"Expected exactly {SEQUENCE_FRAMES} frames."), 400
    try:
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.shape != (SEQUENCE_FRAMES, FEATURES_PER_FRAME) or not np.isfinite(arr).all():
            raise ValueError()
    except Exception:
        return jsonify(ok=False, error="Invalid landmark sequence."), 400
    rows = load_rows()
    rows.append({"label": label, "sequence": arr.tolist()})
    with open(SAMPLES_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    return jsonify(ok=True, label=label, count=sum(1 for r in rows if r["label"] == label), total=len(rows))

@app.post("/api/train")
def train():
    global model
    rows = load_rows()
    counts = Counter(r["label"] for r in rows)
    usable = [r for r in rows if counts[r["label"]] >= 5]
    labels = sorted(set(r["label"] for r in usable))
    if len(labels) < 2:
        return jsonify(ok=False, error="Capture at least 5 sequences for each of at least 2 signs before training."), 400
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import joblib
        X = np.asarray([np.asarray(r["sequence"], dtype=np.float32).reshape(-1) for r in usable])
        y = np.asarray([r["label"] for r in usable])
        # Avoid stratified split failure for very small class counts.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = RandomForestClassifier(
            n_estimators=350, class_weight="balanced_subsample",
            min_samples_leaf=1, max_features="sqrt", random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        score = float(accuracy_score(y_test, clf.predict(X_test)))
        joblib.dump(clf, MODEL_FILE)
        model = clf
        return jsonify(ok=True, accuracy=round(score * 100, 1), samples=len(usable),
                       classes=labels, per_class=dict(counts))
    except Exception as e:
        return jsonify(ok=False, error=f"Training failed: {e}"), 500

@app.post("/api/predict")
def predict():
    if model is None:
        return jsonify(ok=False, error="Train a model first."), 400
    body = request.get_json(silent=True) or {}
    sequence = body.get("sequence")
    try:
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.shape != (SEQUENCE_FRAMES, FEATURES_PER_FRAME) or not np.isfinite(arr).all():
            raise ValueError()
        probs = model.predict_proba([arr.reshape(-1)])[0]
        idx = int(np.argmax(probs))
        return jsonify(ok=True, label=str(model.classes_[idx]), confidence=round(float(probs[idx]), 4))
    except Exception:
        return jsonify(ok=False, error="Invalid sequence."), 400

@app.post("/api/clips")
def upload_clip():
    label = str(request.form.get("label", "")).strip().upper()
    file = request.files.get("clip")
    if not ALLOWED_LABEL.fullmatch(label) or not file or not file.filename:
        return jsonify(ok=False, error="Choose a label and a video file."), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".mp4", ".webm", ".mov"):
        return jsonify(ok=False, error="Use MP4, WEBM or MOV video."), 400
    safe = re.sub(r"[^A-Z0-9_-]+", "_", label)
    filename = f"{safe}{ext}"
    file.save(os.path.join(CLIPS_DIR, filename))
    return jsonify(ok=True, label=label, filename=filename)

@app.get("/api/clips")
def list_clips():
    clips = []
    for filename in sorted(os.listdir(CLIPS_DIR)):
        if filename.lower().endswith((".mp4", ".webm", ".mov")):
            clips.append({"label": os.path.splitext(filename)[0].replace("_", " "),
                          "url": "/sign-clips/" + filename})
    return jsonify(clips=clips)

@app.get("/sign-clips/<path:filename>")
def sign_clip(filename):
    return send_from_directory(CLIPS_DIR, filename)

@app.post("/api/reset")
def reset():
    global model
    for path in (SAMPLES_FILE, MODEL_FILE):
        if os.path.exists(path):
            os.remove(path)
    model = None
    return jsonify(ok=True)

load_model_if_present()

if __name__ == "__main__":
    load_model_if_present()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
