from flask import Flask, flash, render_template, url_for, \
request, redirect

app = Flask(__name__)

@app.route("/", methods = ["GET"])
def index():
    return render_template("index.html")

@app.route("/submit", methods = ["GET", "POST"])
def submit():
    return "hi"
    #return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug = True, port = 8000)

