from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap
from inference import get_prediction
import os

app = Flask(__name__)
Bootstrap(app)

@app.route("/", methods = ["GET", "POST"])
# @app.route("/index")
def index():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file.filename != "":
            image_path = os.path.join("static", uploaded_file.filename)
            uploaded_file.save(image_path)
            class_name = get_prediction(image_path)
            result = {
                "class_name": class_name,
                "image_path": image_path
            }
            return render_template("show.html", result = result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)