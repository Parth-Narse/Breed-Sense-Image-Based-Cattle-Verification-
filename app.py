from flask import Flask, render_template, request
import numpy as np
import cv2
import joblib
import os
import time

app = Flask(__name__)

# Load your model
model = joblib.load("cattle_model.pkl")

# Updated Database: Holstein replaced with Ayrshire + Lifespan added
BREED_INFO = {
    "gir": {
        "region": "Gujarat, India (Saurashtra region)",
        "lifespan": "12–15 years",
        "diseases": "Generally hardy, but susceptible to Mastitis and Foot-and-Mouth Disease (FMD).",
        "maintenance": "Requires moderate grazing. Known for high heat tolerance; provide plenty of water and shade."
    },
    "ponwar": {
        "region": "Uttar Pradesh, India (Pilibhit district)",
        "lifespan": "10–14 years",
        "diseases": "Susceptible to seasonal bacterial infections.",
        "maintenance": "Active and sturdy; requires regular exercise and balanced mineral supplements."
    },
    "ayrshire": {
        "region": "Ayrshire, India",
        "lifespan": "15–20 years",
        "diseases": "Relatively healthy, but watch for typical dairy issues like Ketosis.",
        "maintenance": "Known for being easy to raise; thrives in temperate climates and is an efficient forager."
    }
}

def predict_image(path):
    try:
        img = cv2.imread(path)
        if img is None: return None
        img = cv2.resize(img, (100, 100)) 
        img = img.flatten() / 255.0
        pred = model.predict([img])
        
        if not pred: return None
        return str(pred[0]).lower().strip()
    except:
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            path = "static/temp.jpg"
            file.save(path)
            
            # Image Pre-processing
            img = cv2.imread(path)
            img = cv2.resize(img, (100, 100)).flatten() / 255.0
            
            # GET CONFIDENCE SCORES
            probs = model.predict_proba([img])[0] 
            max_conf = np.max(probs)  # Highest probability
            breed_idx = np.argmax(probs)
            breed_name = model.classes_[breed_idx].lower()

            # TRIGGER ERROR IF CONFIDENCE IS LOW (< 50%)
            if max_conf < 0.70:
                result = "not_identified"
            elif breed_name in BREED_INFO:
                info = BREED_INFO[breed_name]
                result = {
                    "name": breed_name.upper(),
                    "region": info["region"],
                    "lifespan": info["lifespan"],
                    "diseases": info["diseases"],
                    "maintenance": info["maintenance"],
                    "image": path + "?v=" + str(time.time())
                }
            else:
                result = "not_identified"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)