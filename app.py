import os
import logging
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import text
from dotenv import load_dotenv

# OpenAI SDK v1.x
from openai import OpenAI

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

    # Database URL (Render: set in Dashboard -> Environment)
    raw_url = os.environ.get("DATABASE_URL", "sqlite:///l1_jarvis.db")

    # Normalize Postgres URIs for SQLAlchemy + psycopg2 and require SSL on Render
    db_url = raw_url
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    if db_url.startswith("postgresql://") and "+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # Force SSL for Postgres on managed hosts (Render)
    if db_url.startswith("postgresql+psycopg2://"):
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"connect_args": {"sslmode": "require"}}

    # Uploads folder (defaults to /tmp/uploads for cloud)
    upload_folder = os.environ.get("UPLOAD_PATH", "/tmp/uploads")
    os.makedirs(upload_folder, exist_ok=True)
    app.config["UPLOAD_FOLDER"] = upload_folder

    # Allowed file extensions
    app.config["ALLOWED_EXTENSIONS"] = {"txt", "pdf", "csv", "docx"}

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
    name_or_url = db.Column(db.String(255))  # stores either filename or URL label
    filename = db.Column(db.String(255))     # if a file was uploaded
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# -----------------------------
# Helpers
# -----------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def check_database():
    """Try a simple query using the engine so we fail fast if DB is unreachable."""
    try:
        with db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        app.logger.info("Database connection successful")
    except Exception as e:
        app.logger.error(f"Database connection failed: {e}")
        raise RuntimeError("Cannot connect to the database. Check DATABASE_URL / SSL / perms.")


def create_default_admin():
    """Create default admin if the users table is empty."""
    try:
        if User.query.count() == 0:
            admin = User(
                username="admin",
                password_hash=generate_password_hash("admin123"),
            )
            db.session.add(admin)
            db.session.commit()
            app.logger.info("Default admin created: username='admin', password='admin123'")
    except Exception as e:
        app.logger.error(f"Failed creating default admin: {e}")
        raise


def get_openai_client() -> OpenAI:
    """Construct an OpenAI client; logs if key is missing."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        app.logger.warning("OPENAI_API_KEY is not set. AI responses will fail.")
    return OpenAI(api_key=key)


# -----------------------------
# Error Handlers (simple)
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
        company = Company.query.get_or_404(company_id)
        CompanyData.query.filter_by(company_id=company.id).delete()
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
                db.session.add(entry)

            if url_text:
                entry = CompanyData(company_id=company.id, name_or_url=url_text)
                db.session.add(entry)

            db.session.commit()
            flash("Data submitted", "success")
            return redirect(url_for("manage", company_id=company.id))

        data = CompanyData.query.filter_by(company_id=company.id).order_by(CompanyData.created_at.desc()).all()
        return render_template("manage.html", company=company, data=data)

    except Exception as e:
        app.logger.error(f"Manage error: {e}")
        flash("Failed to manage company.", "error")
        return redirect(url_for("dashboard"))


@app.route("/delete_entry/<int:entry_id>")
def delete_entry(entry_id: int):
    if "user_id" not in session:
        return redirect(url_for("login"))

    try:
        entry = CompanyData.query.get_or_404(entry_id)
        if entry.filename:
            try:
                os.remove(os.path.join(app.config["UPLOAD_FOLDER"], entry.filename))
            except FileNotFoundError:
                pass
        db.session.delete(entry)
        db.session.commit()
        flash("Entry deleted", "success")
    except Exception as e:
        app.logger.error(f"Delete entry error: {e}")
        flash("Failed to delete entry.", "error")

    return redirect(request.referrer or url_for("dashboard"))


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

        # Minimal company context: list items (filenames/URLs) to steer the model
        context = ""
        if company_id:
            items = CompanyData.query.filter_by(company_id=company_id)\
                                     .order_by(CompanyData.created_at.desc())\
                                     .limit(20).all()
            if items:
                context = "Company knowledge items:\n" + "\n".join(f"- {it.name_or_url}" for it in items)

        system_prompt = (
            "You are a helpful AI assistant for call center agents. "
            "Answer clearly and concisely based on the provided company knowledge (if any) and the question. "
            "If the answer is not in the provided knowledge, say you don't know."
        )

        user_content = (f"{context}\n\nQuestion:\n{user_message}" if context else user_message)

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=500,
            temperature=0.3,
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
# App startup: check DB, create tables, default admin
# -----------------------------
with app.app_context():
    check_database()
    db.create_all()
    create_default_admin()


# -----------------------------
# Local dev entrypoint
# -----------------------------
if __name__ == "__main__":
    # On Render, use: gunicorn app:app
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
