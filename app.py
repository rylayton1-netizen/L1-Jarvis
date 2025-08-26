import os
import re
import logging
from datetime import datetime
from typing import List, Tuple

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash, abort
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import text, or_
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# OpenAI SDK v1.x
from openai import OpenAI

# Optional fallback for tougher PDFs
try:
    from pdfminer_high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:
    try:
        # Some environments install as pdfminer.high_level
        from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
    except Exception:
        pdfminer_extract_text = None

HAVE_PDFMINER = pdfminer_extract_text is not None

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
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False, index=True)
    email = db.Column(db.String(255), unique=True, nullable=True, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # convenience
    def set_password(self, raw: str):
        self.password_hash = generate_password_hash(raw)

    def check_password(self, raw: str) -> bool:
        return check_password_hash(self.password_hash, raw)


class Company(db.Model):
    __tablename__ = "company"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True, nullable=False)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    owner = db.relationship("User", backref="companies")


class CompanyData(db.Model):
    __tablename__ = "company_data"
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey("company.id"), nullable=False, index=True)
    name_or_url = db.Column(db.String(255))  # filename or URL
    filename = db.Column(db.String(255))     # if a file was uploaded
    content = db.Column(db.Text)             # extracted text
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# -----------------------------
# Retrieval SQL (Step 4)
# -----------------------------
RETRIEVAL_SQL = text("""
WITH
q AS (
  SELECT plainto_tsquery('english', :q) AS tsq
),
k AS (
  SELECT
    'knowledge' AS src,
    k.id,
    k.filename,
    k.source_company_data_id AS source_id,
    k.content,
    NULL::text AS name_or_url,
    ts_rank_cd(cd.content_tsv, q.tsq) AS rank
  FROM public.knowledge k
  JOIN public.company_data cd
    ON cd.id = k.source_company_data_id
  CROSS JOIN q
  WHERE k.company_id = :cid
    AND q.tsq @@ cd.content_tsv
),
d AS (
  SELECT
    'company_data' AS src,
    cd.id,
    cd.filename,
    cd.id AS source_id,
    cd.content,
    cd.name_or_url,
    ts_rank_cd(cd.content_tsv, q.tsq) AS rank
  FROM public.company_data cd
  CROSS JOIN q
  WHERE cd.company_id = :cid
    AND q.tsq @@ cd.content_tsv
),
hits AS (
  SELECT * FROM k
  UNION ALL
  SELECT * FROM d
)
SELECT
  src,
  source_id,
  COALESCE(NULLIF(filename,''), '(no filename)') AS filename,
  name_or_url,
  rank,
  SUBSTRING(content FROM GREATEST(1, POSITION(LOWER(SPLIT_PART(:q, ' ', 1)) IN LOWER(content)) - 120) FOR 280) AS snippet
FROM hits
ORDER BY rank DESC, source_id DESC
LIMIT 8;
""")

def get_context_snippets(question: str, company_id: int):
    """Run retrieval and return a list of dicts [{filename, snippet}, ...]."""
    with db.engine.begin() as conn:
        rows = conn.execute(RETRIEVAL_SQL, {"cid": company_id, "q": question}).fetchall()
    results = []
    for r in rows:
        results.append({
            "filename": r.filename,
            "snippet": (r.snippet or "").strip()
        })
    return results

# -----------------------------
# Guardrails (Step 5)
# -----------------------------
SYSTEM_GUARDRAILS = """You are a call-center assistant constrained to the provided context snippets.
Rules:
- Answer ONLY using the snippets provided.
- If information is insufficient, reply exactly: "I don’t know based on the documents I have."
- Do NOT include filenames, (File: ...), (Source: ...), bracketed citations like [1], or any citation markers in the answer text. The UI will show sources separately.
- Keep answers concise and actionable for agents.
"""

def build_user_prompt(question: str, snippets: List[dict]) -> str:
    lines = []
    for s in snippets:
        lines.append(f"FILE: {s['filename']}\nSNIPPET: {s['snippet']}\n")
    context_block = "\n".join(lines) if lines else "(no context)"
    return f"""Question: {question}

Context snippets:
{context_block}

Instructions:
- Use only the context above.
- Do not include filenames or citation markers in your answer. Return only the answer text.
"""

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
    """
    Creates a default admin user (and a sample company) if DB is empty.
    Safe to run repeatedly.
    """
    try:
        if User.query.count() == 0:
            admin = User(username="admin", email="admin@example.com", password_hash=generate_password_hash("admin123"))
            db.session.add(admin)
            db.session.flush()
            sample_co = Company(name="Admin Co", owner_id=admin.id)
            db.session.add(sample_co)
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
    text_val = ""
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
        text_val = _condense_ws(" ".join(chunks))[:max_chars]
    except Exception as e:
        app.logger.warning(f"PyPDF2 failed for {path}: {e}")

    # Fallback: pdfminer if installed and PyPDF2 result small
    if len(text_val) < 50 and HAVE_PDFMINER:
        try:
            mined = pdfminer_extract_text(path) or ""  # type: ignore
            text_val = _condense_ws(mined)[:max_chars]
            app.logger.info(f"pdfminer extracted {len(text_val)} chars from {os.path.basename(path)}")
        except Exception as e:
            app.logger.error(f"pdfminer failed for {path}: {e}")

    return text_val

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
        for tag in soup(["script", "style"]):
            tag.decompose()
        text_val = soup.get_text(separator=" ")
        out = _condense_ws(text_val)[:max_chars]
        app.logger.info(f"Fetched {len(out)} chars from {url}")
        return out
    except Exception as e:
        app.logger.error(f"URL fetch failed for {url}: {e}")
        return ""

def ingest_entry(entry: CompanyData):
    """Extract text from a file or URL and store in entry.content."""
    text_val = ""
    if entry.filename:
        path = os.path.join(app.config["UPLOAD_FOLDER"], entry.filename)
        ext = entry.filename.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            text_val = extract_text_from_pdf(path)
        elif ext == "txt":
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text_val = _condense_ws(f.read())[:50_000]
            except Exception as e:
                app.logger.error(f"TXT read failed for {path}: {e}")
        elif ext == "csv":
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = []
                    for i, line in enumerate(f):
                        lines.append(line.strip())
                        if i >= 500:
                            break
                    text_val = _condense_ws("\n".join(lines))[:50_000]
            except Exception as e:
                app.logger.error(f"CSV read failed for {path}: {e}")
        else:
            app.logger.info(f"Unsupported file extension for ingestion: {ext}")
    else:
        if entry.name_or_url and entry.name_or_url.lower().startswith(("http://", "https://")):
            text_val = fetch_url_text(entry.name_or_url)

    entry.content = text_val
    db.session.add(entry)
    app.logger.info(f"Ingested item {entry.id or '(new)'} "
                    f"{entry.name_or_url or entry.filename}: {len(text_val)} chars")

# -----------------------------
# Full-text bootstrap (safe to run)
# -----------------------------
def ensure_fulltext_indexes():
    """Create tsvector column/index + triggers (idempotent)."""
    sql = """
    -- Add tsvector on company_data
    ALTER TABLE public.company_data
      ADD COLUMN IF NOT EXISTS content_tsv tsvector;

    -- Backfill missing vectors
    UPDATE public.company_data
    SET content_tsv = to_tsvector('english', coalesce(content,''))
    WHERE content_tsv IS NULL;

    -- Index for fast search
    CREATE INDEX IF NOT EXISTS idx_company_data_tsv ON public.company_data USING GIN (content_tsv);

    -- Trigger to keep it fresh
    CREATE OR REPLACE FUNCTION public.company_data_tsv_refresh()
    RETURNS trigger AS $$
    BEGIN
      NEW.content_tsv := to_tsvector('english', coalesce(NEW.content,''));
      RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

    DROP TRIGGER IF EXISTS trg_company_data_tsv ON public.company_data;
    CREATE TRIGGER trg_company_data_tsv
    BEFORE INSERT OR UPDATE OF content ON public.company_data
    FOR EACH ROW
    EXECUTE FUNCTION public.company_data_tsv_refresh();

    -- Optional: provenance columns on knowledge (filename + link back)
    ALTER TABLE public.knowledge
      ADD COLUMN IF NOT EXISTS filename varchar(255),
      ADD COLUMN IF NOT EXISTS source_company_data_id int REFERENCES public.company_data(id);
    """
    with db.engine.begin() as conn:
        conn.execute(text(sql))

# -----------------------------
# Answer sanitizer
# -----------------------------
def strip_inline_source_suffix(s: str) -> str:
    if not s:
        return s
    lines = s.rstrip().splitlines()
    while lines and re.match(r'^(?:\[\d+\]\s*)?(?:file|source)\s*:\s*.+$', lines[-1].strip(), re.IGNORECASE):
        lines.pop()
    return "\n".join(lines).rstrip()

# -----------------------------
# Auth utilities
# -----------------------------
def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return User.query.get(uid)

def require_login():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return None

def require_company_owner(company_id: int):
    user = current_user()
    if not user:
        return False
    company = Company.query.get_or_404(company_id)
    return (company.owner_id == user.id) or (user.username == "admin")

# 👇 Add this block right after your existing helpers
@app.context_processor
def inject_user():
    """
    Make `current_user`, `is_admin`, and `must_reset_password`
    available in ALL templates automatically.
    """
    user_obj = current_user()
    return {
        "current_user": user_obj,
        "is_admin": bool(user_obj and getattr(user_obj, "username", None) == "admin"),
        "must_reset_password": bool(user_obj and getattr(user_obj, "must_reset_password", False)),
    }


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
# Routes: Auth
# -----------------------------
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    try:
        if request.method == "POST":
            username_or_email = request.form.get("username", "").strip()
            password = request.form.get("password", "")

            user = User.query.filter(
                or_(User.username == username_or_email, User.email == username_or_email)
            ).first()

            if user and user.check_password(password):
                session["user_id"] = user.id
                return redirect(url_for("dashboard"))

            flash("Invalid username/email or password", "error")
        return render_template("login.html")
    except Exception as e:
        app.logger.error(f"Login route error: {e}")
        flash("Login failed. Check logs.", "error")
        return render_template("login.html"), 500

@app.route("/signup", methods=["GET", "POST"])
def signup():
    try:
        if request.method == "POST":
            company_name = (request.form.get("company_name") or "").strip()
            username = (request.form.get("username") or "").strip()
            email = (request.form.get("email") or "").strip().lower() or None
            password = request.form.get("password") or ""
            confirm = request.form.get("confirm") or ""

            # Basic validation
            if not company_name or not username or not password:
                flash("Please fill in all required fields.", "error")
                return render_template("signup.html", preset={
                    "company_name": company_name, "username": username, "email": email or ""
                })
            if password != confirm:
                flash("Passwords do not match.", "error")
                return render_template("signup.html", preset={
                    "company_name": company_name, "username": username, "email": email or ""
                })
            if User.query.filter(or_(User.username == username, User.email == email)).first():
                flash("Username or email is already in use.", "error")
                return render_template("signup.html", preset={
                    "company_name": company_name, "username": username, "email": email or ""
                })
            if Company.query.filter_by(name=company_name).first():
                flash("Company name is taken. Choose another.", "error")
                return render_template("signup.html", preset={
                    "company_name": company_name, "username": username, "email": email or ""
                })

            # Create user + company (atomic)
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.flush()  # get user.id

            company = Company(name=company_name, owner_id=user.id)
            db.session.add(company)
            db.session.commit()

            # Log in
            session["user_id"] = user.id
            flash("Account created! Welcome 👋", "success")
            return redirect(url_for("dashboard"))

        return render_template("signup.html")
    except Exception as e:
        app.logger.error(f"Signup route error: {e}")
        db.session.rollback()
        flash("Sign up failed. Please try again.", "error")
        return render_template("signup.html"), 500

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))

# -----------------------------
# Routes: App
# -----------------------------
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    try:
        user = current_user()
        if request.method == "POST":
            company_name = request.form.get("company_name", "").strip()
            if company_name:
                if Company.query.filter_by(name=company_name).first():
                    flash("Company name is already in use.", "error")
                else:
                    new_company = Company(name=company_name, owner_id=user.id)
                    db.session.add(new_company)
                    db.session.commit()
                    flash("Company created", "success")
                return redirect(url_for("dashboard"))

        # Only show companies owned by this user (admin sees all)
        if user.username == "admin":
            companies = Company.query.order_by(Company.created_at.desc()).all()
        else:
            companies = Company.query.filter_by(owner_id=user.id).order_by(Company.created_at.desc()).all()

        # 👇 THIS is the line you add (instead of the old render_template)
        return render_template("dashboard.html", companies=companies, user=user)
    except Exception as e:
        app.logger.error(f"Dashboard error: {e}")
        flash("Failed to load dashboard.", "error")
        return render_template("dashboard.html", companies=[], user=current_user()), 500

@app.route("/delete_company/<int:company_id>")
def delete_company(company_id: int):
    if "user_id" not in session:
        return redirect(url_for("login"))
    try:
        if not require_company_owner(company_id):
            abort(403)

        # Best-effort cleanup of related rows
        try:
            db.session.execute(text("DELETE FROM knowledge WHERE company_id = :cid"), {"cid": company_id})
        except Exception:
            pass

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
        if not require_company_owner(company_id):
            abort(403)

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
        if not require_company_owner(company_id):
            abort(403)

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
        if not require_company_owner(entry.company_id):
            abort(403)

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
        if not require_company_owner(company_id):
            abort(403)
        company = Company.query.get_or_404(company_id)
        return render_template("agent.html", company=company)
    except Exception as e:
        app.logger.error(f"Agent route error: {e}")
        flash("Failed to load agent page.", "error")
        return redirect(url_for("dashboard"))

@app.route("/agent_api", methods=["POST"])
def agent_api():
    """
    Answers a question for a given company_id by:
    1) Running high-precision retrieval (Postgres FTS) to get filename+snippets
    2) Calling the LLM with strict guardrails to prevent wrong citations
    """
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    try:
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        company_id = int(request.args.get("company_id", "0") or 0)
        if not user_message:
            return jsonify({"answer": "No input received"})

        # authz
        if not require_company_owner(company_id):
            return jsonify({"error": "Forbidden"}), 403

        client = get_openai_client()

        # Build retrieval context (filenames + snippets)
        snippets = []
        if company_id:
            # lazy ensure FTS infra is present
            ensure_fulltext_indexes()
            snippets = get_context_snippets(user_message, company_id)

        # Guardrails + prompt
        user_prompt = build_user_prompt(user_message, snippets)

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_GUARDRAILS},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            temperature=0.2,
        )

        text_out = (resp.choices[0].message.content or "").strip()
        text_out = strip_inline_source_suffix(text_out)

        return jsonify({"answer": text_out, "sources": snippets})
    except Exception as e:
        app.logger.error(f"Agent API error: {e}")
        return jsonify({"answer": f"Error: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/ingest_entry/<int:entry_id>", methods=["POST"])
def ingest_single(entry_id: int):
    if "user_id" not in session:
        return redirect(url_for("login"))
    try:
        entry = CompanyData.query.get_or_404(entry_id)
        if not require_company_owner(entry.company_id):
            abort(403)

        ingest_entry(entry)
        db.session.commit()
        flash(f"Re-ingested entry {entry_id}", "success")
        return redirect(url_for("manage", company_id=entry.company_id))
    except Exception as e:
        app.logger.error(f"Single reingest error: {e}")
        flash("Failed to re-ingest entry.", "error")
        return redirect(url_for("dashboard"))

# -----------------------------
# App startup
# -----------------------------
with app.app_context():
    check_database()
    db.create_all()
    # safe to call repeatedly; creates/refreshes FTS infra
    ensure_fulltext_indexes()
    create_default_admin()

if __name__ == "__main__":
    # On Render, use: gunicorn app:app
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
