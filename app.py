# ── app.py  (stand‑alone, friendly display names) ─────────────────────
import os, numpy as np
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---------------- CONFIG ------------------------------------------------
BASE_DIR        = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER   = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = r"C:\Users\kishore2004\pollen\Flask\model_boost_v2.keras"



ALLOWED_EXT     = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- MODEL -------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded  →  {MODEL_PATH}")
INPUT_H, INPUT_W = model.input_shape[1:3]

# ---------------- CLASS LIST (must match training order) ---------------
CLASS_NAMES = [
    "anadenanthera","arecaceae","cecropia","chromolaena","combretum",
    "croton","dipteryx","eucalipto","faramea","hyptis",
    "mabea","matayba","mimosa","myrcia","protium",
    "qualea","schinus","senegalia","serjania","syagrus","tridax"
]
idx2folder = {i: lbl for i, lbl in enumerate(CLASS_NAMES)}

# ---------------- Friendly display mapping -----------------------------
folder2nice = {
    "anadenanthera": "Anadenanthera colubrina",
    "arecaceae"   : "Arecaceae (Palm pollen)",
    "cecropia"    : "Cecropia spp.",
    "chromolaena" : "Chromolaena odorata",
    "combretum"   : "Combretum leprosum",
    "croton"      : "Croton floribundus",
    "dipteryx"    : "Dipteryx alata",
    "eucalipto"   : "Eucalyptus spp.",
    "faramea"     : "Faramea occidentalis",
    "hyptis"      : "Hyptis suaveolens",
    "mabea"       : "Mabea fistulifera",
    "matayba"     : "Matayba guianensis",
    "mimosa"      : "Mimosa caesalpiniaefolia",
    "myrcia"      : "Myrcia multiflora",
    "protium"     : "Protium heptaphyllum",
    "qualea"      : "Qualea grandiflora",
    "schinus"     : "Schinus terebinthifolia",
    "senegalia"   : "Senegalia polyphylla",
    "serjania"    : "Serjania erecta",
    "syagrus"     : "Syagrus romanzoffiana",
    "tridax"      : "Tridax procumbens"
}

# ---------------- FLASK -------------------------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed(fname):
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def preprocess(path):
    img = load_img(path, target_size=(INPUT_H, INPUT_W))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, 0)

# ---- Routes -----------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("index.html", error="Please choose an image.")

    file = request.files["file"]
    if not allowed(file.filename):
        return render_template("index.html", error="Allowed types: png / jpg / jpeg")

    # Save upload
    fname = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(save_path)

    # Inference
    probs = model.predict(preprocess(save_path))[0]
    idx   = int(np.argmax(probs))
    folder_label = idx2folder[idx]
    display_name = folder2nice.get(folder_label, folder_label)
    conf_pct     = float(probs[idx]) * 100

    result = {
        "prediction": display_name,
        "confidence_numeric": round(conf_pct, 2),
        "image_path": url_for("static", filename=f"uploads/{fname}")
    }
    return render_template("prediction.html", result=result)

# -----------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)


