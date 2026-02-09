from flask import Flask, render_template, request, send_file, redirect, url_for
import os
from detector import process_image
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)

        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], filename)

        file.save(input_path)

        # Process image using OpenCV
        process_image(input_path, output_path)

        return render_template(
            "index.html",
            processed_image=f"outputs/{filename}"
        )

    return render_template("index.html", processed_image=None)


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5000

    print("\n" + "=" * 50)
    print("🚀 Coin Detection Flask Server Started")
    print(f"🌐 Local URL : http://{host}:{port}")
    print("⛔ Press CTRL + C to stop the server")
    print("=" * 50 + "\n")

    app.run(host=host, port=port, debug=True)
