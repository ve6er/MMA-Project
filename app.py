from flask import Flask, jsonify, render_template

app = Flask(__name__)

@app.route("/")
def root():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    print("i got called")
    return jsonify({"score_percentage": 80, "success": True, "score_normalized": 80})

if __name__ == "__main__":
    app.run(debug=True)