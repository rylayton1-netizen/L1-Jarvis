import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import openai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

db = SQLAlchemy(app)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))

class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True)
    data_entries = db.relationship('CompanyData', backref='company', cascade="all, delete-orphan")

class CompanyData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name_or_url = db.Column(db.String(500))
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'))

db.create_all()

# ----------------- Routes -----------------

@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        name = request.form["company_name"].strip()
        if name:
            company = Company(name=name)
            db.session.add(company)
            db.session.commit()
        return redirect(url_for("dashboard"))

    companies = Company.query.all()
    return render_template("dashboard.html", companies=companies)

@app.route("/delete_company/<int:company_id>")
def delete_company(company_id):
    if 'user_id' not in session:
        return redirect(url_for("login"))
    company = Company.query.get_or_404(company_id)
    db.session.delete(company)
    db.session.commit()
    return redirect(url_for("dashboard"))

@app.route("/manage/<int:company_id>", methods=["GET", "POST"])
def manage(company_id):
    if 'user_id' not in session:
        return redirect(url_for("login"))
    company = Company.query.get_or_404(company_id)

    if request.method == "POST":
        file = request.files.get("file")
        url = request.form.get("url", "").strip()
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            entry = CompanyData(name_or_url=filename, company_id=company.id)
            db.session.add(entry)
        if url:
            entry = CompanyData(name_or_url=url, company_id=company.id)
            db.session.add(entry)
        db.session.commit()
        return redirect(url_for("manage", company_id=company.id))

    data = CompanyData.query.filter_by(company_id=company.id).all()
    return render_template("manage.html", company=company, data=data)

@app.route("/delete_entry/<int:entry_id>")
def delete_entry(entry_id):
    if 'user_id' not in session:
        return redirect(url_for("login"))
    entry = CompanyData.query.get_or_404(entry_id)
    company_id = entry.company_id
    db.session.delete(entry)
    db.session.commit()
    return redirect(url_for("manage", company_id=company_id))

# ----------------- Agent Assistant -----------------

@app.route("/agent/<int:company_id>")
def agent(company_id):
    if 'user_id' not in session:
        return redirect(url_for("login"))
    company = Company.query.get_or_404(company_id)
    return render_template("agent.html", company=company)

@app.route("/agent/<int:company_id>/agent_api", methods=["POST"])
def agent_api(company_id):
    if 'user_id' not in session:
        return jsonify({"error":"Unauthorized"}), 401

    data = request.get_json()
    message = data.get("message", "")

    # Call OpenAI API
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"You are a call center assistant for company ID {company_id}. Respond to: {message}. Provide an answer and script separately.",
            temperature=0.7,
            max_tokens=250
        )
        text = response.choices[0].text.strip().split("\n")
        answer = text[0] if len(text) > 0 else "Sorry, no answer."
        script = text[1] if len(text) > 1 else "No script available."
    except Exception as e:
        answer = "Sorry, no response."
        script = "No script available."

    return jsonify({"answer": answer, "script": script})

# ----------------- Run App -----------------
if __name__ == "__main__":
    app.run(debug=True)
