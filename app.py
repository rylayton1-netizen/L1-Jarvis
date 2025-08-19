import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "change_this")
DATABASE_URL = os.getenv("DATABASE_URL")  # e.g. postgresql://user:pass@host:port/dbname

if not OPENAI_API_KEY or not SECRET_KEY or not DATABASE_URL:
    raise ValueError("OPENAI_API_KEY, SECRET_KEY, and DATABASE_URL must be set in .env")

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# --------------------
# DATABASE MODELS
# --------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)

class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)

class CompanyData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=False)
    name_or_url = db.Column(db.String(500), nullable=False)

db.create_all()

# --------------------
# LOGIN ROUTES
# --------------------
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session["user_id"] = user.id
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password", "error")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# --------------------
# DASHBOARD ROUTE
# --------------------
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        company_name = request.form.get("company_name")
        if company_name:
            existing = Company.query.filter_by(name=company_name).first()
            if not existing:
                new_company = Company(name=company_name)
                db.session.add(new_company)
                db.session.commit()
            else:
                flash("Company already exists", "error")

    companies = Company.query.all()
    return render_template("dashboard.html", companies=companies)

@app.route("/delete_company/<int:company_id>")
def delete_company(company_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    company = Company.query.get_or_404(company_id)
    CompanyData.query.filter_by(company_id=company.id).delete()
    db.session.delete(company)
    db.session.commit()
    return redirect(url_for("dashboard"))

# --------------------
# MANAGE ROUTE
# --------------------
@app.route("/manage/<int:company_id>", methods=["GET", "POST"])
def manage(company_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    company = Company.query.get_or_404(company_id)
    if request.method == "POST":
        # File upload
        uploaded_file = request.files.get("file")
        if uploaded_file:
            filename = secure_filename(uploaded_file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(path)
            new_data = CompanyData(company_id=company.id, name_or_url=filename)
            db.session.add(new_data)
        
        # URL submission
        url = request.form.get("url")
        if url:
            new_data = CompanyData(company_id=company.id, name_or_url=url)
            db.session.add(new_data)

        db.session.commit()

    data = CompanyData.query.filter_by(company_id=company.id).all()
    return render_template("manage.html", company=company, data=data)

@app.route("/delete_entry/<int:entry_id>")
def delete_entry(entry_id):
    entry = CompanyData.query.get_or_404(entry_id)
    db.session.delete(entry)
    db.session.commit()
    return redirect(request.referrer or url_for("dashboard"))

# --------------------
# AGENT ASSISTANT
# --------------------
@app.route("/agent/<int:company_id>")
def agent(company_id):
    company = Company.query.get_or_404(company_id)
    return render_template("agent.html", company_id=company.id)

@app.route("/agent_api", methods=["POST"])
def agent_api():
    data = request.get_json()
    user_msg = data.get("message")
    if not user_msg:
        return jsonify({"answer": "No message received", "script": ""})

    # OpenAI GPT call
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_msg}]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Error: {e}"

    # Simple dynamic script (can be replaced with GPT logic)
    script = f"Suggested action for: {user_msg}"

    return jsonify({"answer": answer, "script": script})

# --------------------
# RUN
# --------------------
if __name__ == "__main__":
    app.run(debug=True)
