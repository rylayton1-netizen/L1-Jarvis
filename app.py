import os, io
from datetime import datetime
from urllib.parse import urlparse

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
from dotenv import load_dotenv
from openai import OpenAI

import requests
from bs4 import BeautifulSoup
import pandas as pd
from PyPDF2 import PdfReader

# ---------- ENV & APP ----------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///instance/multi_agent.db")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "password")

if not OPENAI_API_KEY or not SECRET_KEY or not DATABASE_URL:
    raise ValueError("OPENAI_API_KEY, SECRET_KEY, and DATABASE_URL must be set in .env")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = SECRET_KEY

# ---------- DB ----------
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ---------- OPENAI ----------
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- MODELS ----------
class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(160), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Knowledge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey("company.id"), nullable=False, index=True)
    source_type = db.Column(db.String(16), nullable=False)  
    source_name = db.Column(db.String(512), nullable=True)  
    content = db.Column(db.Text, nullable=False)
    embedding = db.Column(db.PickleType, nullable=True)  # store vector as pickle
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ---------- HELPERS ----------
def login_required(view_func):
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper

def is_valid_url(u: str) -> bool:
    try:
        parts = urlparse(u)
        return parts.scheme in ("http", "https") and parts.netloc != ""
    except Exception:
        return False

def extract_text_from_pdf(file_storage) -> str:
    data = file_storage.read()
    reader = PdfReader(io.BytesIO(data))
    texts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(texts).strip()

def extract_text_from_csv(file_storage) -> str:
    try:
        df = pd.read_csv(file_storage)
    except Exception:
        file_storage.seek(0)
        df = pd.read_csv(file_storage, encoding="utf-8-sig", engine="python")
    return df.to_string(index=False)

def crawl_url(url: str, timeout=15) -> str:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "L1-KB-Crawler/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    lines = [ln.strip() for ln in soup.get_text().splitlines()]
    return "\n".join([ln for ln in lines if ln])[:150000]

def compute_embedding(text: str):
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

def retrieve_relevant_knowledge(company_id, query, top_k=3):
    """Semantic search by embedding similarity"""
    embedding = compute_embedding(query)
    items = Knowledge.query.filter_by(company_id=company_id).all()
    scored = []
    for k in items:
        if not k.embedding:
            continue
        # cosine similarity
        vec = k.embedding
        score = sum([a*b for a,b in zip(vec, embedding)]) / (max(1e-8, (sum([a*a for a in vec])**0.5)*(sum([a*a for a in embedding])**0.5)))
        scored.append((score, k.content))
    scored.sort(reverse=True)
    return "\n".join([content for _, content in scored[:top_k]]) if scored else ""

# ---------- ROUTES ----------
@app.route("/")
def root():
    return redirect(url_for("dashboard") if session.get("user") else url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method=="POST":
        u = request.form.get("username","").strip()
        p = request.form.get("password","")
        if u==ADMIN_USER and p==ADMIN_PASS:
            session["user"]=u
            return redirect(url_for("dashboard"))
        error="Invalid username/password"
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    companies = Company.query.order_by(Company.created_at.desc()).all()
    return render_template("dashboard.html", companies=companies)

@app.route("/create_company", methods=["POST"])
@login_required
def create_company():
    name = (request.form.get("name") or "").strip()
    if name:
        try:
            db.session.add(Company(name=name))
            db.session.commit()
            flash("Company created", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"Error: {e}", "danger")
    else:
        flash("Company name required", "danger")
    return redirect(url_for("dashboard"))

@app.route("/delete_company/<int:company_id>", methods=["POST"])
@login_required
def delete_company(company_id):
    try:
        Knowledge.query.filter_by(company_id=company_id).delete()
        Company.query.filter_by(id=company_id).delete()
        db.session.commit()
        flash("Company deleted","success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error: {e}","danger")
    return redirect(url_for("dashboard"))

@app.route("/manage/<int:company_id>")
@login_required
def manage(company_id):
    company = Company.query.get_or_404(company_id)
    items = Knowledge.query.filter_by(company_id=company_id).order_by(Knowledge.created_at.desc()).all()
    return render_template("manage.html", company=company, knowledge=items)

@app.route("/manage/<int:company_id>/add_manual", methods=["POST"])
@login_required
def add_manual(company_id):
    company = Company.query.get_or_404(company_id)
    content = (request.form.get("manual_content") or "").strip()
    if content:
        emb = compute_embedding(content)
        db.session.add(Knowledge(company_id=company.id, source_type="manual", source_name="Manual", content=content, embedding=emb))
        db.session.commit()
        flash("Manual knowledge added","success")
    return redirect(url_for("manage", company_id=company_id))

@app.route("/manage/<int:company_id>/add_url", methods=["POST"])
@login_required
def add_url(company_id):
    company = Company.query.get_or_404(company_id)
    url = (request.form.get("url") or "").strip()
    if not is_valid_url(url):
        flash("Invalid URL","danger")
        return redirect(url_for("manage", company_id=company_id))
    try:
        text = crawl_url(url)
        emb = compute_embedding(text)
        db.session.add(Knowledge(company_id=company.id, source_type="url", source_name=url, content=text, embedding=emb))
        db.session.commit()
        flash("URL added and embedded","success")
    except Exception as e:
        flash(f"Error: {e}","danger")
    return redirect(url_for("manage", company_id=company_id))

@app.route("/manage/<int:company_id>/add_file", methods=["POST"])
@login_required
def add_file(company_id):
    company = Company.query.get_or_404(company_id)
    file = request.files.get("file")
    if not file: flash("Select a file","danger"); return redirect(url_for("manage", company_id=company_id))
    fname = secure_filename(file.filename)
    ext = fname.lower().rsplit(".",1)[-1]
    if ext=="pdf": content = extract_text_from_pdf(file)
    elif ext=="csv": content = extract_text_from_csv(file)
    else: flash("Only PDF/CSV supported","danger"); return redirect(url_for("manage", company_id=company_id))
    emb = compute_embedding(content)
    db.session.add(Knowledge(company_id=company.id, source_type="file", source_name=fname, content=content, embedding=emb))
    db.session.commit()
    flash(f"File {fname} added","success")
    return redirect(url_for("manage", company_id=company_id))

@app.route("/manage/delete_knowledge/<int:item_id>", methods=["POST"])
@login_required
def delete_knowledge(item_id):
    item = Knowledge.query.get_or_404(item_id)
    company_id = item.company_id
    db.session.delete(item)
    db.session.commit()
    flash("Knowledge deleted","success")
    return redirect(url_for("manage", company_id=company_id))

@app.route("/agent/<int:company_id>")
def agent(company_id):
    company = Company.query.get_or_404(company_id)
    return render_template("agent.html", company=company)

@app.route("/chat/<int:company_id>", methods=["POST"])
def chat(company_id):
    company = Company.query.get_or_404(company_id)
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    context = retrieve_relevant_knowledge(company.id, user_message, top_k=3)
    prompt = f"""
You are a phone assistant for "{company.name}".
Use knowledge context if helpful. Today: {datetime.utcnow().strftime('%B %d, %Y')}

User Query:
{user_message}

Knowledge Context:
{context}

Respond in format:
Answer:
<answer>

Script:
<concise phone script>
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=500
        )
        text = resp.choices[0].message.content.strip()
        return jsonify({"ok": True, "text": text})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__=="__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)), debug=True)
