import os
import cv2
from flask import Flask, render_template, request
from detector import detect_coins

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        if "image" not in request.files:
            return "No file uploaded", 400

        file = request.files["image"]

        if file.filename == "":
            return "No selected file", 400

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        output_img, table = detect_coins(input_path)

        output_filename = "result.png"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, output_img)

        return render_template(
            "index.html",
            output_image=output_filename,
            table=table
        )

    return render_template("index.html")


if __name__ == "__main__":
    print("Server running at http://127.0.0.1:5000")
    app.run(debug=True)
