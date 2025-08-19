import os
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
import logging

# --- Setup ---
load_dotenv()  # Load environment variables from .env

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
logger = logging.getLogger(__name__)

# --- Models ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    schema_name = db.Column(db.String(120), unique=True, nullable=False)

# --- Forms ---
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

# --- Login ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Routes ---
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
        schema_name = form.name.data.replace(" ", "_").lower()
        company = Company(name=form.name.data, schema_name=schema_name)
        try:
            db.session.add(company)
            db.session.commit()
            # Create knowledge table for this company
            table_query = f"""
                CREATE TABLE IF NOT EXISTS knowledge_{schema_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT
                )
            """
            db.session.execute(table_query)
            db.session.commit()
            flash("Company created successfully", "success")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating company: {str(e)}")
            flash(f"Error creating company: {str(e)}", "error")
    companies = Company.query.all()
    return render_template("dashboard.html", form=form, companies=companies)

@app.route("/delete_company/<int:company_id>", methods=["POST"])
@login_required
def delete_company(company_id):
    company = Company.query.get_or_404(company_id)
    try:
        db.session.delete(company)
        db.session.commit()
        flash(f"Deleted company {company.name}", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting company: {str(e)}", "error")
    return redirect(url_for("dashboard"))

@app.route("/manage/<int:company_id>", methods=["GET", "POST"])
@login_required
def manage(company_id):
    company = Company.query.get_or_404(company_id)
    form = UploadForm()
    form.company_id.choices = [(c.id, c.name) for c in Company.query.all()]
    form.company_id.data = company_id

    if form.validate_on_submit():
        schema = f"knowledge_{company.schema_name}"
        try:
            with db.engine.connect() as conn:
                if form.file.data:
                    file = form.file.data
                    content = ""
                    if file.filename.endswith(".csv"):
                        df = pd.read_csv(file)
                        content = df.to_string()
                    elif file.filename.endswith(".pdf"):
                        reader = PdfReader(file)
                        content = "\n".join([page.extract_text() for page in reader.pages])
                    conn.execute(f"INSERT INTO {schema} (content) VALUES (:content)", {"content": content})

                if form.url.data:
                    response = requests.get(form.url.data)
                    soup = BeautifulSoup(response.text, "html.parser")
                    content = soup.get_text(strip=True)
                    conn.execute(f"INSERT INTO {schema} (content) VALUES (:content)", {"content": content})

            flash("Data added successfully", "success")
        except Exception as e:
            logger.error(f"Error uploading data: {str(e)}")
            flash(f"Error uploading data: {str(e)}", "error")

    return render_template("manage.html", form=form, company=company)

@app.route("/agent/<company_name>")
def agent(company_name):
    company = Company.query.filter_by(name=company_name).first_or_404()
    return render_template("index.html", company_name=company_name)

@app.route("/query/<company_name>", methods=["POST"])
def query(company_name):
    company = Company.query.filter_by(name=company_name).first_or_404()
    query_text = request.form.get("query", "").strip()
    if not query_text:
        return jsonify({"answer": "Error: Query cannot be empty"})

    schema = f"knowledge_{company.schema_name}"
    try:
        with db.engine.connect() as conn:
            result = conn.execute(f"SELECT content FROM {schema} WHERE content LIKE :q", {"q": f"%{query_text}%"})
            results = result.fetchall()
            context = " ".join([row[0] for row in results]) if results else "No relevant information found."

        prompt = (
            f"Based on this information: {context}\n\n"
            f"Current date: August 19, 2025. Answer the question for a phone support agent "
            f"handling inquiries about {company_name} emergency medications: {query_text}\n"
            "Then, provide a professional script to say on the phone, prefixed with 'Script:'."
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return jsonify({"answer": f"Error: {str(e)}"})

# --- Initialize DB ---
with app.app_context():
    db.create_all()
    if not User.query.first():
        admin = User(username="admin", password="password")
        db.session.add(admin)
        db.session.commit()

# --- Run App ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
