import os
import re
import logging
from datetime import datetime
from typing import List, Tuple

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import text
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# OpenAI SDK v1.x
from openai import OpenAI

# Optional fallback for tougher PDFs
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.layout import LAParams
    HAVE_PDFMINER = True
except Exception:
    HAVE_PDFMINER = False

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# App Factory / Config
# -----------------------------
def create_app():
    app = Flask(__name__)

    # Secret key
    app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")

    # Database URL
    raw_url = os.environ.get("DATABASE_URL", "sqlite:///l1_jarvis.db")

    # Normalize Postgres URIs for SQLAlchemy + psycopg2 and require SSL on Render
    db_url = raw_url
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    if db_url.startswith("postgresql://") and "+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Force SSL for managed Postgres
    if db_url.startswith("postgresql+psycopg2://"):
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"connect_args": {"sslmode": "require"}}

    # Uploads folder (prefer /tmp/uploads on Render)
    upload_folder = os.environ.get("UPLOAD_PATH", "/tmp/uploads")
    os.makedirs(upload_folder, exist_ok=True)
    app.config["UPLOAD_FOLDER"] = upload_folder

    # Allowed file extensions
    app.config["ALLOWED_EXTENSIONS"] = {"txt", "pdf", "csv"}

    # Logging
    logging.basicConfig(level=logging.INFO)
    app.logger.info(f"DATABASE_URL (normalized): {db_url}")
    app.logger.info(f"UPLOAD_FOLDER: {upload_folder}")

    return app

app = create_app()
db = SQLAlchemy(app)

# -----------------------------
# Database Models
# -----------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class CompanyData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey("company.id"), nullable=False)
    name_or_url = db.Column(db.String(255))  # filename or URL
    filename = db.Column(db.String(255))     # if a file was uploaded
    content = db.Column(db.Text)             # extracted text
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# -----------------------------
# Helpers
# -----------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def check_database():
    try:
        with db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        app.logger.info("Database connection successful")
    except Exception as e:
        app.logger.error(f"Database connection failed: {e}")
        raise RuntimeError("Cannot connect to DB. Check DATABASE_URL / SSL / perms.")

def create_default_admin():
    try:
        if User.query.count() == 0:
            admin = User(username="admin", password_hash=generate_password_hash("admin123"))
            db.session.add(admin)
            db.session.commit()
            app.logger.info("Default admin created: admin / admin123")
    except Exception as e:
        app.logger.error(f"Failed creating default admin: {e}")
        raise

def get_openai_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        app.logger.warning("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key)

def _condense_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def extract_text_from_pdf(path: str, max_chars: int = 50_000) -> str:
    """
    Try PyPDF2 first; if we get too little text (scanned PDF or tricky layout),
    fall back to pdfminer.six which is more robust for text extraction.
    """
    text = ""
    # Attempt 1: PyPDF2
    try:
        reader = PdfReader(path)
        chunks = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                chunks.append(t)
            if sum(len(c) for c in chunks) >= max_chars:
                break
        text = _condense_ws(" ".join(chunks))[:max_chars]
    except Exception as e:
        app.logger.warning(f"PyPDF2 failed for {path}: {e}")

    # Fallback: pdfminer if installed and PyPDF2 result small
    if len(text) < 50 and HAVE_PDFMINER:
        try:
            laparams = LAParams()
            mined = pdfminer_extract_text(path, laparams=laparams) or ""
            text = _condense_ws(mined)[:max_chars]
            app.logger.info(f"pdfminer extracted {len(text)} chars from {os.path.basename(path)}")
        except Exception as e:
            app.logger.error(f"pdfminer failed for {path}: {e}")
            text = text  # keep whatever we had (likely empty)

    return text

def fetch_url_text(url: str, max_chars: int = 50_000) -> str:
    try:
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/122.0.0.0 Safari/537.36")
        }
        r = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Strip obvious noise but keep <noscript> (some sites render fallback content there)
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        out = _condense_ws(text)[:max_chars]
        app.logger.info(f"Fetched {len(out)} chars from {url}")
        return out
    except Exception as e:
        app.logger.error(f"URL fetch failed for {url}: {e}")
        return ""

def ingest_entry(entry: CompanyData):
    """Extract text from a file or URL and store in entry.content."""
    text = ""
    if entry.filename:
        path = os.path.join(app.config["UPLOAD_FOLDER"], entry.filename)
        ext = entry.filename.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            text = extract_text_from_pdf(path)
        elif ext == "txt":
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = _condense_ws(f.read())[:50_000]
            except Exception as e:
                app.logger.error(f"TXT read failed for {path}: {e}")
        elif ext == "csv":
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = []
                    for i, line in enumerate(f):
                        lines.append(line.strip())
                        if i >= 500:  # grab a decent chunk
                            break
                    text = _condense_ws("\n".join(lines))[:50_000]
            except Exception as e:
                app.logger.error(f"CSV read failed for {path}: {e}")
        else:
            app.logger.info(f"Unsupported file extension for ingestion: {ext}")
    else:
        # Treat as URL if it looks like one
        if entry.name_or_url and entry.name_or_url.lower().startswith(("http://", "https://")):
            text = fetch_url_text(entry.name_or_url)

    entry.content = text
    db.session.add(entry)
    app.logger.info(f"Ingested item {entry.id or '(new)'} "
                    f"{entry.name_or_url or entry.filename}: {len(text)} chars")

def simple_rank(items: List[CompanyData], query: str, k: int = 5) -> List[CompanyData]:
    """Basic keyword scoring to pick the most relevant entries."""
    q = query.lower()
    terms = [t for t in re.split(r"[^a-z0-9]+", q) if t]
    if not terms:
        return items[:k]

    ranked: List[Tuple[int, CompanyData]] = []
    for it in items:
        if not it.content:
            continue
        text_low = it.content.lower()
        score = 0
        for t in terms:
            score += text_low.count(t) * 5
        score += len({t for t in terms if t in text_low})
        if score > 0:
            ranked.append((score, it))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in ranked[:k]]

def smart_snippets(text: str, query: str, window: int = 600, max_snippets: int = 3) -> str:
    """Return a few merged windows around matches, or a head slice if no match."""
    if not text:
        return ""
    low = text.lower()
    terms = [t for t in re.split(r"[^a-z0-9]+", (query or "").lower()) if t]
    if not terms:
        return _condense_ws(text[: 2 * window * max_snippets])

    spans = []
    for t in terms:
        start = 0
        while True:
            i = low.find(t, start)
            if i == -1:
                break
            s = max(0, i - window)
            e = min(len(text), i + len(t) + window)
            spans.append((s, e))
            start = i + len(t)
    if not spans:
        return _condense_ws(text[: 2 * window * max_snippets])

    spans.sort()
    merged = []
    cs, ce = spans[0]
    for s, e in spans[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    merged = merged[:max_snippets]
    chunks = [_condense_ws(text[s:e]) for s, e in merged]
    return "\n...\n".join(chunks)

# -----------------------------
# Errors
# -----------------------------
@app.errorhandler(404)
def not_found(_):
    return render_template("login.html"), 404

@app.errorhandler(500)
def server_error(_):
    flash("An unexpected error occurred. Please try again.", "error")
    return render_template("login.html"), 500

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    try:
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")

            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password_hash, password):
                session["user_id"] = user.id
                return redirect(url_for("dashboard"))

            flash("Invalid username or password", "error")
        return render_template("login.html")
    except Exception as e:
        app.logger.error(f"Login route error: {e}")
        flash("Login failed. Check logs.", "error")
        return render_template("login.html"), 500

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    try:
        if request.method == "POST":
            company_name = request.form.get("company_name", "").strip()
            if company_name:
                new_company = Company(name=company_name)
                db.session.add(new_company)
                db.session.commit()
                flash("Company created", "success")
                return redirect(url_for("dashboard"))
        companies = Company.query.order_by(Company.created_at.desc()).all()
        return render_template("dashboard.html", companies=companies)
    except Exception as e:
        app.logger.error(f"Dashboard error: {e}")
        flash("Failed to load dashboard.", "error")
        return render_template("dashboard.html", companies=[]), 500

@app.route("/delete_company/<int:company_id>")
def delete_company(company_id: int):
    if "user_id" not in session:
        return redirect(url_for("login"))
    try:
        # Remove related rows in legacy 'knowledge' table if it exists
        try:
            db.session.execute(text("DELETE FROM knowledge WHERE company_id = :cid"), {"cid": company_id})
        except Exception:
            pass  # table may not exist

        CompanyData.query.filter_by(company_id=company_id).delete()
        company = Company.query.get_or_404(company_id)
        db.session.delete(company)
        db.session.commit()
        flash("Company deleted", "success")
    except Exception as e:
        app.logger.error(f"Delete company error: {e}")
        flash("Failed to delete company.", "error")
    return redirect(url_for("dashboard"))

@app.route("/manage/<int:company_id>", methods=["GET", "POST"])
def manage(company_id: int):
    if "user_id" not in session:
        return redirect(url_for("login"))
    try:
        company = Company.query.get_or_404(company_id)

        if request.method == "POST":
            file = request.files.get("file")
            url_text = request.form.get("url", "").strip()

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(save_path)
                entry = CompanyData(company_id=company.id, filename=filename, name_or_url=filename)
                ingest_entry(entry)

            if url_text:
                entry = CompanyData(company_id=company.id, name_or_url=url_text)
                ingest_entry(entry)

            db.session.commit()
            flash("Data ingested", "success")
            return redirect(url_for("manage", company_id=company.id))

        data = CompanyData.query.filter_by(company_id=company.id).order_by(CompanyData.created_at.desc()).all()
        return render_template("manage.html", company=company, data=data)
    except Exception as e:
        app.logger.error(f"Manage error: {e}")
        flash("Failed to manage company.", "error")
        return redirect(url_for("dashboard"))

@app.route("/reingest/<int:company_id>")
def reingest(company_id: int):
    if "user_id" not in session:
        return redirect(url_for("login"))
    try:
        company = Company.query.get_or_404(company_id)
        items = CompanyData.query.filter_by(company_id=company.id).all()
        for it in items:
            ingest_entry(it)
        db.session.commit()
        flash("Re-ingested all items.", "success")
    except Exception as e:
        app.logger.error(f"Reingest error: {e}")
        flash("Failed to re-ingest.", "error")
    return redirect(url_for("manage", company_id=company_id))

@app.route("/delete_entry/<int:entry_id>", methods=["GET"])
def delete_entry(entry_id: int):
    if "user_id" not in session:
        return redirect(url_for("login"))
    try:
        entry = CompanyData.query.get_or_404(entry_id)
        company_id = entry.company_id
        if entry.filename:
            try:
                path = os.path.join(app.config["UPLOAD_FOLDER"], entry.filename)
                if os.path.exists(path):
                    os.remove(path)
            except (FileNotFoundError, PermissionError):
                pass
        db.session.delete(entry)
        db.session.commit()
        flash("Entry deleted", "success")
        return redirect(url_for("manage", company_id=company_id))
    except Exception as e:
        app.logger.error(f"Delete entry error: {e}")
        flash("Failed to delete entry.", "error")
        return redirect(url_for("dashboard"))

@app.route("/agent/<int:company_id>")
def agent(company_id: int):
    if "user_id" not in session:
        return redirect(url_for("login"))
    try:
        company = Company.query.get_or_404(company_id)
        return render_template("agent.html", company=company)
    except Exception as e:
        app.logger.error(f"Agent route error: {e}")
        flash("Failed to load agent page.", "error")
        return redirect(url_for("dashboard"))

@app.route("/agent_api", methods=["POST"])
def agent_api():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        company_id = int(request.args.get("company_id", "0") or 0)
        if not user_message:
            return jsonify({"answer": "No input received"})

        client = get_openai_client()

        # Collect and rank company content
        context = ""
        if company_id:
            items = CompanyData.query.filter_by(company_id=company_id).order_by(CompanyData.created_at.desc()).all()

            # Lazily ingest any empty items
            updated = False
            for it in items:
                if not it.content:
                    try:
                        ingest_entry(it)
                        updated = True
                    except Exception as e:
                        app.logger.error(f"Lazy ingest failed for item {it.id}: {e}")
            if updated:
                db.session.commit()

            # Build generous context with smart snippets
            pieces = []
            remaining = 6000  # char budget for context
            for it in simple_rank(items, user_message, k=5):
                text = it.content or ""
                if not text:
                    continue
                title = it.name_or_url or it.filename or "Item"
                if len(text) <= 3500:
                    chunk = text
                else:
                    chunk = smart_snippets(text, user_message, window=600, max_snippets=3)
                block = f"### {title}\n{chunk}"
                if len(block) > remaining:
                    block = block[:remaining]
                pieces.append(block)
                remaining -= len(block)
                if remaining <= 0:
                    break
            context = "\n\n".join(pieces)
            app.logger.info(f"Agent context items={len(pieces)} chars={len(context)}")

        system_prompt = (
            "You are a helpful AI assistant for call center agents. "
            "Answer clearly and concisely using ONLY the provided context. "
            "If the context does not contain the answer, say you don't know "
            "and suggest which document or page to check."
        )
        user_content = user_message if not context else f"Context:\n{context}\n\nQuestion:\n{user_message}"

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=500,
            temperature=0.2,
        )

        text_out = resp.choices[0].message.content.strip()
        return jsonify({"answer": text_out})
    except Exception as e:
        app.logger.error(f"Agent API error: {e}")
        return jsonify({"answer": f"Error: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

# -----------------------------
# App startup
# -----------------------------
with app.app_context():
    check_database()
    db.create_all()
    create_default_admin()

if __name__ == "__main__":
    # On Render, use: gunicorn app:app
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
