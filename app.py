from flask import Flask, render_template, request, jsonify
import face_recognition
import numpy as np
import pickle
import base64
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)
DATA_FILE = "students.pkl"

# Load or init database
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        students = pickle.load(f)
else:
    students = {}

def decode_image(data_url):
    header, encoded = data_url.split(",", 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_data))
    return np.array(image)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["POST"])
def register():
    name = request.form["name"]
    roll = request.form["roll"]
    dept = request.form["dept"]
    college = request.form["college"]
    image_data = request.form["image"]

    image_np = decode_image(image_data)
    faces = face_recognition.face_encodings(image_np)

    if not faces:
        return "❌ No face detected", 400

    students[name] = {
        "encoding": faces[0],
        "roll": roll,
        "dept": dept,
        "college": college
    }

    with open(DATA_FILE, "wb") as f:
        pickle.dump(students, f)

    return "✅ Student registered!"

@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.get_json()
    image_np = decode_image(data["image"])
    unknown_faces = face_recognition.face_encodings(image_np)

    if not unknown_faces:
        return jsonify({"status": "❌ No face detected"})

    unknown = unknown_faces[0]

    for name, info in students.items():
        match = face_recognition.compare_faces([info["encoding"]], unknown)[0]
        if match:
            return jsonify({
                "status": "✅ Match Found!",
                "name": name,
                "roll": info["roll"],
                "dept": info["dept"],
                "college": info["college"]
            })

    return jsonify({"status": "❌ No match found"})

if __name__ == "__main__":
    app.run(debug=True)
