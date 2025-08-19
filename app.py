import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import openai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not all([OPENAI_API_KEY, SECRET_KEY, DATABASE_URL]):
    raise ValueError("OPENAI_API_KEY, SECRET_KEY, and DATABASE_URL must be set in .env")

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------- Models ----------------
class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True, nullable=False)
    url = db.Column(db.String(300), nullable=True)

class Knowledge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "password":
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    companies = Company.query.all()
    if request.method == "POST":
        name = request.form.get("company_name")
        url = request.form.get("company_url")
        if name:
            company = Company(name=name, url=url)
            db.session.add(company)
            db.session.commit()
            flash("Company created successfully", "success")
        return redirect(url_for("dashboard"))

    return render_template("dashboard.html", companies=companies)

@app.route("/delete_company/<int:company_id>")
def delete_company(company_id):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    company = Company.query.get_or_404(company_id)
    db.session.delete(company)
    db.session.commit()
    flash("Company deleted successfully", "success")
    return redirect(url_for("dashboard"))

@app.route("/manage/<int:company_id>", methods=["GET", "POST"])
def manage(company_id):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    company = Company.query.get_or_404(company_id)
    if request.method == "POST":
        # Add files or URLs (example, simple text submission)
        content = request.form.get("content")
        if content:
            kb = Knowledge(company_id=company.id, content=content)
            db.session.add(kb)
            db.session.commit()
            flash("Content added successfully", "success")
    knowledge = Knowledge.query.filter_by(company_id=company.id).all()
    return render_template("manage.html", company=company, knowledge=knowledge)

@app.route("/agent/<int:company_id>", methods=["GET"])
def agent(company_id):
    company = Company.query.get_or_404(company_id)
    return render_template("index.html", company=company)

@app.route("/ask/<int:company_id>", methods=["POST"])
def ask(company_id):
    company = Company.query.get_or_404(company_id)
    question = request.json.get("question")
    knowledge_entries = Knowledge.query.filter_by(company_id=company.id).all()
    context = "\n".join([k.content for k in knowledge_entries])
    prompt = f"Answer the question based on the following knowledge:\n{context}\nQuestion: {question}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=300
    )
    answer = response.choices[0].text.strip()
    script = f"Script for: {question}"  # replace with actual dynamic generation
    return jsonify({"answer": answer, "script": script})

# ---------------- Main ----------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
