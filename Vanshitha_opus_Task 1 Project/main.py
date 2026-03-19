from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
import numpy as np
from flask_mysqldb import MySQL
import json

app = Flask(__name__)

# =========================
# MySQL CONFIG
# =========================
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Prakash@27'
app.config['MYSQL_DB'] = 'healthcare'   # your DB name

mysql = MySQL(app)

# =========================
# Load Model
# =========================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# TEMP USER STORAGE
user_data = {}

# =========================
# PAGE 1
# =========================
@app.route("/")
def home():
    return render_template("user.html")

@app.route("/user", methods=["POST"])
def user():
    global user_data
    user_data = {
        "name": request.form["name"],
        "age": request.form["age"],
        "gender": request.form["gender"]
    }
    return redirect(url_for("predict_page"))

# =========================
# PAGE 2
# =========================
@app.route("/predict_page")
def predict_page():
    return render_template("predict.html", user=user_data)

# =========================
# AJAX PREDICTION
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        features = data["features"]

        arr = np.array(features).reshape(1, -1)
        arr = scaler.transform(arr)

        prediction = int(model.predict(arr)[0])
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        # SAVE TO MYSQL
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO predictions (name, age, gender, input_data, prediction)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            data["name"],
            data["age"],
            data["gender"],
            json.dumps(features),
            prediction
        ))
        mysql.connection.commit()
        cur.close()

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})

# =========================
if __name__ == "__main__":
    app.run(debug=True)