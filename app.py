import os
import re
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, SelectField
from wtforms.validators import DataRequired, URL
from dotenv import load_dotenv
import pandas as pd
from PyPDF2 import PdfReader
import validators
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

# ---------- Setup ----------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not OPENAI_API_KEY or not SECRET_KEY or not DATABASE_URL:
    raise ValueError("OPENAI_API_KEY, SECRET_KEY, and DATABASE_URL must be set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("l1")

# ---------- Models ----------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    schema_name = db.Column(db.String(120), unique=True, nullable=False)

# ---------- Forms ----------
class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

class CompanyForm(FlaskForm):
    name = StringField("Company Name", validators=[DataRequired()])
    submit = SubmitField("Create Company")

class UploadForm(FlaskForm):
    company_id = SelectField("Select Company", choices=[], coerce=int)
    file = FileField("Upload CSV/PDF")
    url = StringField("Crawl URL", validators=[URL(require_tld=True, message="Invalid URL")])
    submit = SubmitField("Submit")

# ---------- Helpers ----------
def sanitize_schema_name(raw: str) -> str:
    """lowercase, convert spaces to _, remove anything not a-z0-9_"""
    s = raw.strip().lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    if not s:
        s = "company"
    return s

def table_name_for(company: Company) -> str:
    # single flat table per company: knowledge_<schema_name>
    safe = sanitize_schema_name(company.schema_name)
    return f"knowledge_{safe}"

def ensure_company_table_exists(company: Company):
    tbl = table_name_for(company)
    create_sql = text(f"""
        CREATE TABLE IF NOT EXISTS {tbl} (
            id SERIAL PRIMARY KEY,
            content TEXT
        )
    """)
    with db.engine.begin() as conn:
        conn.execute(create_sql)

# ---------- Login ----------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------- Routes ----------
@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.password == form.password.data:
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "error")
    return render_template("login.html", form=form)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    form = CompanyForm()
    if form.validate_on_submit():
        schema_name = sanitize_schema_name(form.name.data)
        company = Company(name=form.name.data.strip(), schema_name=schema_name)
        try:
            db.session.add(company)
            db.session.commit()
            ensure_company_table_exists(company)
            flash("Company created successfully", "success")
        except IntegrityError:
            db.session.rollback()
            flash("Company name already exists", "error")
        except Exception as e:
            db.session.rollback()
            logger.exception("Error creating company")
            flash(f"Error creating company: {e}", "error")

    companies = Company.query.order_by(Company.name.asc()).all()
    return render_template("dashboard.html", form=form, companies=companies)

@app.route("/delete_company/<int:company_id>", methods=["POST"])
@login_required
def delete_company(company_id):
    company = Company.query.get_or_404(company_id)
    # Optionally drop its knowledge table
    try:
        tbl = table_name_for(company)
        with db.engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {tbl}"))
        db.session.delete(company)
        db.session.commit()
        flash(f"Deleted company {company.name}", "success")
    except Exception as e:
        db.session.rollback()
        logger.exception("Error deleting company")
        flash(f"Error deleting company: {e}", "error")
    return redirect(url_for("dashboard"))

@app.route("/manage/<int:company_id>", methods=["GET", "POST"])
@login_required
def manage(company_id):
    company = Company.query.get_or_404(company_id)
    ensure_company_table_exists(company)

    form = UploadForm()
    form.company_id.choices = [(c.id, c.name) for c in Company.query.order_by(Company.name.asc()).all()]
    form.company_id.data = company_id

    if form.validate_on_submit():
        tbl = table_name_for(company)
        try:
            with db.engine.begin() as conn:

                # Handle file upload
                if form.file.data:
                    file = form.file.data
                    content = ""
                    if file.filename.endswith(".csv"):
                        df = pd.read_csv(file)
                        content = df.to_string(index=False)
                    elif file.filename.endswith(".pdf"):
                        reader = PdfReader(file)
                        content = "\n".join([(p.extract_text() or "") for p in reader.pages])
                    if content and content.strip():
                        conn.execute(text(f"INSERT INTO {tbl} (content) VALUES (:content)"), {"content": content})
                        flash("File content added successfully", "success")
                    else:
                        flash("Uploaded file contains no extractable text.", "warning")

                # Handle URL crawl
                if form.url.data:
                    if not validators.url(form.url.data):
                        flash("Invalid URL format", "error")
                    else:
                        r = requests.get(form.url.data, timeout=15)
                        r.raise_for_status()
                        soup = BeautifulSoup(r.text, "html.parser")
                        page_text = soup.get_text(separator=" ", strip=True)
                        if page_text and page_text.strip():
                            conn.execute(text(f"INSERT INTO {tbl} (content) VALUES (:content)"), {"content": page_text})
                            flash("URL content added successfully", "success")
                        else:
                            flash("No readable text found at URL.", "warning")

        except Exception as e:
            logger.exception("Error processing upload/crawl")
            flash(f"Error processing upload/crawl: {e}", "error")

    return render_template("manage.html", form=form, company=company)

@app.route("/agent/<company_name>")
def agent(company_name):
    company = Company.query.filter_by(name=company_name).first_or_404()
    ensure_company_table_exists(company)
    return render_template("index.html", company_name=company_name)

@app.route("/query/<company_name>", methods=["POST"])
def query(company_name):
    company = Company.query.filter_by(name=company_name).first_or_404()
    ensure_company_table_exists(company)

    user_query = request.form.get("query", "").strip()
    if not user_query:
        return jsonify({"answer": "Error: Query cannot be empty"})

    tbl = table_name_for(company)

    try:
        # Postgres: ILIKE for case-insensitive search
        like_param = f"%{user_query}%"
        with db.engine.begin() as conn:
            rows = conn.execute(
                text(f"SELECT content FROM {tbl} WHERE content ILIKE :q"),
                {"q": like_param}
            ).fetchall()

        context = " ".join([r[0] for r in rows]) if rows else "No relevant information found."

        prompt = (
            f"Based on this information: {context}\n\n"
            f"Answer the question for a phone support agent handling inquiries about {company.name}: {user_query}\n"
            f"Then, provide a professional script to say on the phone, prefixed with 'Script:'."
        )

        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = resp.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        logger.exception("Query error")
        return jsonify({"answer": f"Error: {e}"})

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

# ---------- Init DB ----------
with app.app_context():
    db.create_all()
    if not User.query.first():
        admin = User(username="admin", password="password")
        db.session.add(admin)
        db.session.commit()
