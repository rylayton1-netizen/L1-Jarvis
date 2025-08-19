import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import openai
from dotenv import load_dotenv

load_dotenv()

# ----------------- Config -----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not all([OPENAI_API_KEY, SECRET_KEY, DATABASE_URL]):
    raise ValueError("OPENAI_API_KEY, SECRET_KEY, and DATABASE_URL must be set in .env")

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ----------------- Models -----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password_hash = db.Column(db.String(200))

class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True)
    data = db.relationship("CompanyData", backref="company", cascade="all, delete-orphan")

class CompanyData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey("company.id"))
    name_or_url = db.Column(db.String(255))
    content = db.Column(db.Text)

# ----------------- Database Creation -----------------
with app.app_context():
    db.create_all()

# ----------------- Routes -----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session["user_id"] = user.id
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        company_name = request.form["company_name"]
        if company_name:
            existing = Company.query.filter_by(name=company_name).first()
            if not existing:
                new_company = Company(name=company_name)
                db.session.add(new_company)
                db.session.commit()
    companies = Company.query.all()
    return render_template("dashboard.html", companies=companies)

@app.route("/delete_company/<int:company_id>")
def delete_company(company_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    company = Company.query.get_or_404(company_id)
    db.session.delete(company)
    db.session.commit()
    return redirect(url_for("dashboard"))

@app.route("/manage/<int:company_id>", methods=["GET", "POST"])
def manage(company_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    company = Company.query.get_or_404(company_id)
    if request.method == "POST":
        file = request.files.get("file")
        url_input = request.form.get("url")
        if file:
            content = file.read().decode("utf-8")
            new_entry = CompanyData(company=company, name_or_url=file.filename, content=content)
            db.session.add(new_entry)
        if url_input:
            new_entry = CompanyData(company=company, name_or_url=url_input, content="")
            db.session.add(new_entry)
        db.session.commit()
    data = CompanyData.query.filter_by(company_id=company_id).all()
    return render_template("manage.html", company=company, data=data)

@app.route("/delete_entry/<int:entry_id>")
def delete_entry(entry_id):
    entry = CompanyData.query.get_or_404(entry_id)
    db.session.delete(entry)
    db.session.commit()
    return redirect(url_for("manage", company_id=entry.company_id))

@app.route("/agent/<int:company_id>")
def agent(company_id):
    company = Company.query.get_or_404(company_id)
    return render_template("agent.html", company=company)

@app.route("/agent_api", methods=["POST"])
def agent_api():
    data = request.get_json()
    message = data.get("message")
    if not message:
        return jsonify({"answer": "No message sent.", "script": ""})
    
    # Call OpenAI API
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Company AI assistant. User: {message}\nProvide Answer and Script separated clearly.",
            temperature=0.7,
            max_tokens=300
        )
        text = response.choices[0].text.strip()
        # Split into Answer and Script if possible
        if "Script:" in text:
            answer, script = text.split("Script:", 1)
        else:
            answer, script = text, ""
        return jsonify({"answer": answer.strip(), "script": script.strip()})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}", "script": ""})

# ----------------- Run -----------------
if __name__ == "__main__":
    app.run(debug=True)
