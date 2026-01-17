from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("acid_base_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        ph = float(request.form["ph"])
        temp = float(request.form["temp"])
        conc = float(request.form["conc"])

        prediction = model.predict([[ph, temp, conc]])
        result = "Basic" if prediction[0] == 1 else "Acidic"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
