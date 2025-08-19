import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret")
DATABASE_URL = os.getenv("DATABASE_URL")

if not OPENAI_API_KEY or not DATABASE_URL:
    raise ValueError("OPENAI_API_KEY and DATABASE_URL must be set in .env")

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# ---- Routes ---- #

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "password":
            session["user"] = username
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    db = SessionLocal()
    if request.method == "POST":
        company = request.form["company"]
        try:
            db.execute(text(f"CREATE SCHEMA IF NOT EXISTS knowledge_{company}"))
            db.execute(text(f"""
                CREATE TABLE IF NOT EXISTS knowledge_{company}.knowledge (
                    id SERIAL PRIMARY KEY,
                    content TEXT
                )
            """))
            db.commit()
        except Exception as e:
            db.rollback()
            return render_template("dashboard.html", companies=get_companies(), error=f"Error creating company: {e}")
    return render_template("dashboard.html", companies=get_companies())

@app.route("/manage/<company>", methods=["GET", "POST"])
def manage(company):
    if "user" not in session:
        return redirect(url_for("login"))

    db = SessionLocal()
    if request.method == "POST":
        content = request.form.get("content")
        if content:
            try:
                db.execute(text(f"INSERT INTO knowledge_{company}.knowledge (content) VALUES (:c)"), {"c": content})
                db.commit()
            except Exception as e:
                db.rollback()
                return render_template("manage.html", company=company, error=f"Error: {e}")
    return render_template("manage.html", company=company)

@app.route("/agent/<company>")
def agent(company):
    return render_template("agent.html", company=company)

@app.route("/chat/<company>", methods=["POST"])
def chat(company):
    user_message = request.json.get("message")

    db = SessionLocal()
    result = db.execute(
        text(f"SELECT content FROM knowledge_{company}.knowledge WHERE content LIKE :q"),
        {"q": f"%{user_message}%"}
    ).fetchall()
    db.close()

    knowledge = "\n".join([row[0] for row in result]) if result else "No relevant data found."

    prompt = f"""
    You are an AI call assistant. Use the knowledge base when possible.

    User: {user_message}
    Knowledge Base: {knowledge}

    Respond with the following format:

    Answer: <direct helpful answer to the user>

    Script: <a suggested script the agent should read>
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        ai_reply = response["choices"][0]["message"]["content"]
    except Exception as e:
        ai_reply = f"Error: {str(e)}"

    return jsonify({"reply": ai_reply})

def get_companies():
    db = SessionLocal()
    result = db.execute(text("SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'knowledge_%'")).fetchall()
    db.close()
    return [r[0].replace("knowledge_", "") for r in result]

if __name__ == "__main__":
    app.run(debug=True)
