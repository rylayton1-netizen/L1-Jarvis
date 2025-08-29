import os
import re
import logging
import datetime as dt
from datetime import datetime
from typing import List, Optional

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash, make_response
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import text, asc, or_, bindparam
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import jwt  # PyJWT

# OpenAI SDK v1.x
from openai import OpenAI

# Register pgvector adapters for SQLAlchemy bindings (lists -> vector)
try:
    from pgvector.sqlalchemy import Vector  # noqa: F401
except Exception:
    Vector = None

# Optional fallback for tougher PDFs
try:
    from pdfminer_high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:
    try:
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

    # Normalize Postgres URIs for SQLAlchemy + psycopg2 and require SSL on hosted
    db_url = raw_url
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    if db_url.startswith("postgresql://") and "+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Force SSL for managed Postgres (no-op for sqlite)
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
# JWT (Embed Mode)
# -----------------------------
JWT_SECRET = os.environ.get("EMBED_JWT_SECRET", "change-me")
JWT_ALGO = "HS256"

def make_embed_token(
    company_id: int,
    campaign_id: Optional[int] = None,
    agent_label: str = "callshaper",
    ttl_hours: Optional[int] = 8
) -> str:
    """
    Mint a JWT for iframe embed. If ttl_hours == 0, token has NO 'exp' (non-expiring).
    """
    now = dt.datetime.utcnow()
    payload = {
        "sub": "agent-embed",
        "company_id": company_id,
        "campaign_id": campaign_id,
        "agent_label": agent_label,
        "scopes": ["agent_view", "agent_actions"],
        "iat": now,
    }
    try:
        th = int(ttl_hours) if ttl_hours is not None else 8
    except (TypeError, ValueError):
        th = 8
    if th > 0:
        payload["exp"] = now + dt.timedelta(hours=th)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def verify_token_from_request():
    token = request.args.get("token")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth.split(" ", 1)[1].strip()
    if not token:
        raise jwt.InvalidTokenError("Missing token")
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])

def try_get_embed_claims_silent():
    try:
        return verify_token_from_request()
    except Exception:
        return None

def embed_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            claims = verify_token_from_request()
        except Exception:
            return jsonify({"error": "unauthorized"}), 401
        request.embed_claims = claims
        return f(*args, **kwargs)
    return wrapper

# -----------------------------
# Database Models
# -----------------------------
class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    must_reset_password = db.Column(db.Boolean, default=False, nullable=False)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Company(db.Model):
    __tablename__ = "company"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    owner_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)


class CompanyData(db.Model):
    __tablename__ = "company_data"
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey("company.id"), nullable=False)
    name_or_url = db.Column(db.String(255))  # filename or URL
    filename = db.Column(db.String(255))     # if a file was uploaded
    content = db.Column(db.Text)             # extracted text
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ---------------------------------------------------------------------
# (Legacy) FTS retrieval SQL used previously. Keeping for reference.
# ---------------------------------------------------------------------
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
# Guardrails
# -----------------------------
SYSTEM_GUARDRAILS = """You are a call-center agent assistant constrained to the provided context.
Rules:
- Answer ONLY using the provided context blocks.
- If information is insufficient, reply exactly: "I don’t know based on the information I have."
- Do NOT include filenames, (File: ...), (Source: ...), bracketed citations like [1], or any citation markers in the answer text. The UI will show sources separately.
- Keep answers concise and actionable for agents.
"""

# -----------------------------
# Embeddings / Hybrid Retrieval
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"  # 1536 dims (matches DB vector(1536))
TOP_K_VEC = 8
TOP_K_FTS = 8
FINAL_K = 6
EMBED_DIM = 1536  # keep in one place

def embed_query_text(client: OpenAI, text_in: str):
    """Return a 1536-dim vector for the query using OpenAI embeddings."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=text_in)
    return resp.data[0].embedding

def ensure_knowledge_tsv_fresh(company_id: Optional[int] = None):
    """
    Populate public.knowledge.tsv for rows with NULL/empty tsv.
    Safe/idempotent; call at startup and before queries.
    """
    sql = """
    UPDATE public.knowledge
    SET tsv = to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,''))
    WHERE (tsv IS NULL OR tsv = ''::tsvector)
    {cid_filter};
    """.format(cid_filter="AND company_id = :cid" if company_id else "")
    with db.engine.begin() as conn:
        conn.execute(text(sql), ({"cid": company_id} if company_id else {}))

def hybrid_retrieve_from_knowledge(company_id: int, query_text: str, client: OpenAI):
    """
    Returns list of dicts: {id, title, url, location, content, sim}
    Combines vector ANN (embedding <=> :qvec) with FTS (ts_rank), merges, sorts.
    """
    # 1) embed query
    qvec = embed_query_text(client, query_text)

    # 2) vector hits
    if Vector is not None:
        # Clean, typed binding: ensures :qvec is a true pgvector vector(1536)
        vec_sql = text("""
            SELECT id, company_id, title, url, location, content,
                   1 - (embedding <=> :qvec) AS sim
            FROM public.knowledge
            WHERE company_id = :cid AND embedding IS NOT NULL
            ORDER BY embedding <=> :qvec
            LIMIT :k
        """).bindparams(bindparam("qvec", type_=Vector(EMBED_DIM)))
        vector_params = {"cid": company_id, "qvec": qvec, "k": TOP_K_VEC}
    else:
        # Fallback: explicit cast to vector if pgvector.sqlalchemy is unavailable
        vec_sql = text("""
            SELECT id, company_id, title, url, location, content,
                   1 - (embedding <=> :qvec::vector) AS sim
            FROM public.knowledge
            WHERE company_id = :cid AND embedding IS NOT NULL
            ORDER BY embedding <=> :qvec::vector
            LIMIT :k
        """)
        vector_params = {"cid": company_id, "qvec": qvec, "k": TOP_K_VEC}

    # 3) FTS hits
    fts_sql = text("""
        SELECT id, company_id, title, url, location, content,
               ts_rank(tsv, plainto_tsquery('english', :q)) AS sim
        FROM public.knowledge
        WHERE company_id = :cid
          AND tsv @@ plainto_tsquery('english', :q)
        ORDER BY sim DESC
        LIMIT :k
    """)

    with db.engine.begin() as conn:
        vhits = conn.execute(vec_sql, vector_params).mappings().all()
        fhits = conn.execute(fts_sql, {"cid": company_id, "q": query_text, "k": TOP_K_FTS}).mappings().all()

    # 4) merge by id; keep max sim
    merged = {}
    for r in list(vhits) + list(fhits):
        rid = r["id"]
        if rid not in merged or float(r["sim"]) > float(merged[rid]["sim"]):
            merged[rid] = dict(r)

    # 5) sort and top-K
    candidates = sorted(merged.values(), key=lambda x: float(x["sim"]), reverse=True)
    return candidates[:FINAL_K]

def build_user_prompt(question: str, chunks: List[dict]) -> str:
    """
    chunks expects dictionaries with keys: title, url, location, content
    """
    blocks = []
    for i, c in enumerate(chunks, start=1):
        title = c.get("title") or "(untitled)"
        url = c.get("url") or "(no url)"
        quote = (c.get("content") or "").strip()
        loc = c.get("location")
        head = f"[{i}] {title} @ {url}" + (f" • {loc}" if loc else "")
        blocks.append(f"{head}\n{quote}\n")
    context_block = "\n".join(blocks) if blocks else "(no context)"

    return f"""Question:
{question}

Use ONLY the context below. If the answer isn’t clearly supported, respond exactly:
"I don’t know based on the information I have."

Context:
{context_block}
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

def ensure_schema_upgrades():
    """
    Bring an older DB up to date without Alembic. Safe to run repeatedly.
    """
    ddl = """
    -- user.email
    ALTER TABLE "user" ADD COLUMN IF NOT EXISTS email varchar(255);
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = 'ix_user_email' AND n.nspname = 'public'
      ) THEN
        CREATE UNIQUE INDEX ix_user_email ON "user"(email) WHERE email IS NOT NULL;
      END IF;
    END $$;

    -- user.must_reset_password
    ALTER TABLE "user" ADD COLUMN IF NOT EXISTS must_reset_password boolean DEFAULT false NOT NULL;

    -- company.owner_id
    ALTER TABLE company ADD COLUMN IF NOT EXISTS owner_id integer;
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_company_owner'
      ) THEN
        ALTER TABLE company
          ADD CONSTRAINT fk_company_owner
          FOREIGN KEY (owner_id) REFERENCES "user"(id) ON DELETE SET NULL;
      END IF;
    END $$;
    CREATE INDEX IF NOT EXISTS ix_company_owner_id ON company(owner_id);

    -- optional provenance on knowledge table
    ALTER TABLE public.knowledge
      ADD COLUMN IF NOT EXISTS filename varchar(255),
      ADD COLUMN IF NOT EXISTS source_company_data_id int REFERENCES public.company_data(id);
    """
    with db.engine.begin() as conn:
        conn.execute(text(ddl))

def create_default_admin():
    try:
        if User.query.count() == 0:
            admin = User(
                username="admin",
                email="admin@example.com",
                password_hash=generate_password_hash("admin123", method="pbkdf2:sha256", salt_length=16),
            )
            db.session.add(admin)
            db.session.flush()
            sample_co = Company(name="Admin Co", owner_id=admin.id)
            db.session.add(sample_co)
            db.session.commit()
            app.logger.info("Default admin created: admin / admin123")
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed creating default admin: {e}")

def get_openai_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        app.logger.warning("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key)

def _condense_ws(text_in: str) -> str:
    return re.sub(r"\s+", " ", text_in or "").strip()

def extract_text_from_pdf(path: str, max_chars: int = 50_000) -> str:
    text_val = ""
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
    """Create tsvector column/index + triggers (idempotent) on company_data; provenance on knowledge."""
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
# Error handlers
# -----------------------------
@app.errorhandler(404)
def not_found(_):
    return render_template("login.html"), 404

@app.errorhandler(500)
def server_error(_):
    flash("An unexpected error occurred. Please try again.", "error")
    return render_template("login.html"), 500

# -----------------------------
# Auth utilities + context processor
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

@app.context_processor
def inject_user():
    user_obj = current_user()
    return {
        "current_user": user_obj,
        "is_admin": bool(user_obj and getattr(user_obj, "username", None) == "admin"),
        "must_reset_password": bool(user_obj and getattr(user_obj, "must_reset_password", False)),
    }

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
                if getattr(user, "must_reset_password", False):
                    flash("Please set a new password to continue.", "error")
                    return redirect(url_for("change_password"))
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

            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.flush()

            company = Company(name=company_name, owner_id=user.id)
            db.session.add(company)
            db.session.commit()

            session["user_id"] = user.id
            flash("Account created! Welcome 👋", "success")
            return redirect(url_for("dashboard"))

        return render_template("signup.html")
    except Exception as e:
        app.logger.error(f"Signup route error: {e}")
        db.session.rollback()
        flash("Sign up failed. Please try again.", "error")
        return render_template("signup.html"), 500

@app.route("/change_password", methods=["GET", "POST"])
def change_password():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = current_user()
    if request.method == "POST":
        current = request.form.get("current_password", "")
        new = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")

        if not user.check_password(current):
            flash("Current password is incorrect.", "error")
            return render_template("change_password.html")

        if len(new) < 8:
            flash("New password must be at least 8 characters.", "error")
            return render_template("change_password.html")

        if new != confirm:
            flash("Passwords do not match.", "error")
            return render_template("change_password.html")

        user.password_hash = generate_password_hash(new, method="pbkdf2:sha256", salt_length=16)
        if hasattr(user, "must_reset_password"):
            user.must_reset_password = False
        db.session.add(user)
        db.session.commit()

        flash("Password updated successfully.", "success")
        return redirect(url_for("dashboard"))

    return render_template("change_password.html")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))

# -----------------------------
# Routes: Admin
# -----------------------------
def _require_admin():
    u = current_user()
    return bool(u and u.username == "admin")

@app.route("/admin/users", methods=["GET"])
def admin_users():
    if not _require_admin():
        flash("Forbidden", "error")
        return redirect(url_for("dashboard"))
    users = User.query.order_by(asc(User.username)).all()
    return render_template("admin_users.html", users=users)

@app.route("/admin/reset_password/<int:user_id>", methods=["POST"])
def admin_reset_password(user_id: int):
    if not _require_admin():
        flash("Forbidden", "error")
        return redirect(url_for("dashboard"))

    me = current_user()
    if me and me.id == user_id:
        flash("Use your own change-password page for your account.", "error")
        return redirect(url_for("admin_users"))

    user = User.query.get_or_404(user_id)
    user.password_hash = generate_password_hash("Temp1234", method="pbkdf2:sha256", salt_length=16)
    if hasattr(user, "must_reset_password"):
        user.must_reset_password = True
    db.session.add(user)
    db.session.commit()
    flash(f"Password for '{user.username}' reset to Temp1234.", "success")
    return redirect(url_for("admin_users"))

@app.route("/admin/embed_url/<int:company_id>", methods=["GET"])
def admin_embed_url(company_id: int):
    """
    Helper to mint a tokenized iframe URL. Admins or the company owner only.
    TTL behavior:
      - ttl_hours > 0  -> token expires after that many hours
      - ttl_hours == 0 -> token has NO 'exp' (non-expiring)
    """
    if "user_id" not in session:
        return redirect(url_for("login"))

    company = Company.query.get_or_404(company_id)
    user = current_user()
    if not ((company.owner_id == (user.id if user else None)) or (user and user.username == "admin")):
        return jsonify({"error": "forbidden"}), 403

    campaign_id = request.args.get("campaign_id")  # optional; dashboard no longer sends it
    ttl_param = request.args.get("ttl_hours", "8")
    try:
        ttl_hours = int(ttl_param)
    except (TypeError, ValueError):
        ttl_hours = 8

    token = make_embed_token(
        company_id=company_id,
        campaign_id=int(campaign_id) if campaign_id else None,
        ttl_hours=ttl_hours
    )
    base = request.url_root.rstrip("/")
    url = f"{base}/agent/embed?token={token}"
    return jsonify({"company_id": company_id, "embed_url": url, "ttl_hours": ttl_hours})

# -----------------------------
# Routes: App (dashboard/companies/agent)
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

        if user.username == "admin":
            companies = Company.query.order_by(Company.created_at.desc()).all()
        else:
            companies = Company.query.filter_by(owner_id=user.id).order_by(Company.created_at.desc()).all()

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
        try:
            db.session.execute(text("DELETE FROM knowledge WHERE company_id = :cid"), {"cid": company_id})
        except Exception:
            pass
        CompanyData.query.filter_by(company_id=company_id).delete()
        company = Company.query.get_or_404(company_id)
        user = current_user()
        if not ((company.owner_id == (user.id if user else None)) or (user and user.username == "admin")):
            flash("Forbidden", "error")
            return redirect(url_for("dashboard"))
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

        user = current_user()
        if not ((company.owner_id == (user.id if user else None)) or (user and user.username == "admin")):
            flash("Forbidden", "error")
            return redirect(url_for("dashboard"))

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
        user = current_user()
        if not ((company.owner_id == (user.id if user else None)) or (user and user.username == "admin")):
            flash("Forbidden", "error")
            return redirect(url_for("dashboard"))

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

        company = Company.query.get_or_404(company_id)
        user = current_user()
        if not ((company.owner_id == (user.id if user else None)) or (user and user.username == "admin")):
            flash("Forbidden", "error")
            return redirect(url_for("dashboard"))

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
        user = current_user()
        if not ((company.owner_id == (user.id if user else None)) or (user and user.username == "admin")):
            flash("Forbidden", "error")
            return redirect(url_for("dashboard"))
        return render_template("agent.html", company=company)
    except Exception as e:
        app.logger.error(f"Agent route error: {e}")
        flash("Failed to load agent page.", "error")
        return redirect(url_for("dashboard"))

# -----------------------------
# Cookie-free embed route for CallShaper iframe
# -----------------------------
@app.route("/agent/embed", methods=["GET"])
@embed_required
def agent_embed():
    claims = getattr(request, "embed_claims", {})
    company_id = claims.get("company_id")
    company = Company.query.get_or_404(company_id)

    resp = make_response(render_template(
        "agent.html",            # reuse the same template for visual consistency
        company=company,
        embed_mode=True          # optional flag if you want it in your template
    ))
    resp.headers["Content-Security-Policy"] = (
        "frame-ancestors 'self' https://manage.callshaper.com https://*.callshaper.com;"
    )
    return resp

# -----------------------------
# Agent API (HYBRID retrieval)
#   - Works with normal session OR with a valid embed token
# -----------------------------
@app.post("/agent_api")
def agent_api():
    """
    Answers a question for a given company_id by:
    1) Hybrid retrieval from public.knowledge (pgvector + FTS) to get high-recall chunks
    2) Calling the LLM with strict guardrails
    3) Returning 'sources' with exact quotes + URLs for the UI

    Auth:
      - Session cookie (normal app), OR
      - Bearer/Query token issued by make_embed_token (embed mode)
    """
    claims = try_get_embed_claims_silent()
    has_session = "user_id" in session

    if not has_session and not claims:
        return jsonify({"answer": None, "sources": [], "error": "unauthorized"}), 401

    try:
        data = request.get_json(silent=True) or {}
        if not data and request.form:
            data = request.form.to_dict()

        user_message = (data.get("message") or "").strip()
        # If token claims exist, force company_id from token. Else accept query param.
        company_id = int(claims["company_id"]) if claims else int(request.args.get("company_id", "0") or 0)
        show_sources = bool(data.get("show_sources", True))

        if not user_message:
            payload = {"answer": None, "sources": [], "error": "no_input"}
            app.logger.info("agent_api: empty user_message; sending %s", payload)
            return jsonify(payload), 400

        client = get_openai_client()

        # Keep knowledge.tsv fresh for this company (safe + idempotent)
        ensure_knowledge_tsv_fresh(company_id if company_id else None)

        # --- HYBRID RETRIEVAL ---
        hits = hybrid_retrieve_from_knowledge(company_id, user_message, client) if company_id else []

        # Build LLM context + UI sources
        llm_chunks: List[dict] = []
        sources_payload: List[dict] = []
        for h in hits:
            llm_chunks.append({
                "title": h.get("title"),
                "url": h.get("url"),
                "location": h.get("location"),
                "content": h.get("content"),
            })
            sources_payload.append({
                "title": h.get("title") or "(untitled)",
                "filename": None,
                "url": h.get("url"),
                "quote": (h.get("content") or ""),
                "location": h.get("location"),
            })

        user_prompt = build_user_prompt(user_message, llm_chunks)

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_GUARDRAILS},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            temperature=0.2,
        )

        text_out = strip_inline_source_suffix((resp.choices[0].message.content or "").strip())

        payload = {
            "answer": text_out,
            "sources": sources_payload if show_sources else [],
            "error": None
        }
        app.logger.info(
            "agent_api OK (len=%d, hits=%d): %s",
            len(text_out or ""), len(hits or []),
            str({**payload, "answer": (text_out[:120] + ('…' if len(text_out) > 120 else ''))})
        )
        return jsonify(payload), 200

    except Exception as e:
        app.logger.exception("Agent API error")
        return jsonify({"answer": None, "sources": [], "error": str(e)}), 500

# -----------------------------
# Minimal data endpoint for embed page boot
# -----------------------------
@app.get("/api/embed/agent_view")
@embed_required
def agent_view_data():
    claims = getattr(request, "embed_claims", {})
    return jsonify({
        "ok": True,
        "company_id": claims.get("company_id"),
        "campaign_id": claims.get("campaign_id"),
        "agent_label": claims.get("agent_label", "agent")
    })

# -----------------------------
# Health + single-ingest route
# -----------------------------
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/ingest_entry/<int:entry_id>", methods=["POST"])
def ingest_single(entry_id: int):
    if "user_id" not in session:
        return redirect(url_for("login"))
    try:
        entry = CompanyData.query.get_or_404(entry_id)

        company = Company.query.get_or_404(entry.company_id)
        user = current_user()
        if not ((company.owner_id == (user.id if user else None)) or (user and user.username == "admin")):
            flash("Forbidden", "error")
            return redirect(url_for("dashboard"))

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
    ensure_schema_upgrades()   # ensure columns/indexes/FK exist
    ensure_fulltext_indexes()  # company_data FTS infra + knowledge provenance
    ensure_knowledge_tsv_fresh(None)  # keep knowledge.tsv populated
    create_default_admin()     # safe if already present

if __name__ == "__main__":
    # On Render, use: gunicorn app:app
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
