import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import openai

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not OPENAI_API_KEY or not SECRET_KEY or not DATABASE_URL:
    raise ValueError("OPENAI_API_KEY, SECRET_KEY, and DATABASE_URL must be set in .env")

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# --- MODELS ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))

class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), unique=True)

class Knowledge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey("company.id"))
    content = db.Column(db.Text)

# Initialize DB within app context
with app.app_context():
    db.create_all()

# --- ROUTES ---

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session["user"] = user.id
            return redirect(url_for("dashboard"))
        else:
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

    if request.method == "POST":
        name = request.form.get("company_name")
        if name:
            try:
                new_company = Company(name=name)
                db.session.add(new_company)
                db.session.commit()
            except Exception as e:
                return f"Error creating company: {e}"

    companies = Company.query.all()
    return render_template("dashboard.html", companies=companies)

@app.route("/delete_company/<int:company_id>")
def delete_company(company_id):
    company = Company.query.get_or_404(company_id)
    db.session.delete(company)
    db.session.commit()
    return redirect(url_for("dashboard"))

@app.route("/manage/<int:company_id>", methods=["GET", "POST"])
def manage(company_id):
    if "user" not in session:
        return redirect(url_for("login"))

    company = Company.query.get_or_404(company_id)
    if request.method == "POST":
        file = request.files.get("file")
        url = request.form.get("url")
        if file:
            content = file.read().decode("utf-8")
            db.session.add(Knowledge(company_id=company.id, content=content))
        if url:
            db.session.add(Knowledge(company_id=company.id, content=url))
        db.session.commit()
    
    data = Knowledge.query.filter_by(company_id=company.id).all()
    return render_template("manage.html", company=company, data=data)

@app.route("/delete_entry/<int:entry_id>")
def delete_entry(entry_id):
    entry = Knowledge.query.get_or_404(entry_id)
    db.session.delete(entry)
    db.session.commit()
    return redirect(url_for("manage", company_id=entry.company_id))

@app.route("/agent/<int:company_id>")
def agent(company_id):
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("agent.html", company_id=company_id)

@app.route("/agent_api", methods=["POST"])
def agent_api():
    data = request.get_json()
    message = data.get("message")
    company_id = data.get("company_id")

    # Fetch knowledge base for the company
    knowledge = Knowledge.query.filter_by(company_id=company_id).all()
    context = "\n".join([k.content for k in knowledge])

    # Call OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Use the following knowledge base to answer questions:\n{context}"},
                {"role": "user", "content": message}
            ],
            max_tokens=300
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Error contacting OpenAI: {e}"

    return jsonify({"answer": answer, "script": "Generated script goes here"})

if __name__ == "__main__":
    app.run(debug=True)
