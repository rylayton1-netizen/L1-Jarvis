import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import openai

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    data = db.relationship('CompanyData', backref='company', cascade='all, delete-orphan')

class CompanyData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=False)
    name_or_url = db.Column(db.String(500), nullable=False)

# Initialize DB
with app.app_context():
    db.create_all()

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        name = request.form['company_name']
        if name:
            db.session.add(Company(name=name))
            db.session.commit()

    companies = Company.query.all()
    return render_template('dashboard.html', companies=companies)

@app.route('/delete_company/<int:company_id>')
def delete_company(company_id):
    company = Company.query.get_or_404(company_id)
    db.session.delete(company)
    db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/manage/<int:company_id>', methods=['GET', 'POST'])
def manage(company_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    company = Company.query.get_or_404(company_id)

    if request.method == 'POST':
        url = request.form.get('url')
        file = request.files.get('file')
        if url:
            db.session.add(CompanyData(company_id=company.id, name_or_url=url))
        if file:
            db.session.add(CompanyData(company_id=company.id, name_or_url=file.filename))
        db.session.commit()

    data = company.data
    return render_template('manage.html', company=company, data=data)

@app.route('/delete_entry/<int:entry_id>')
def delete_entry(entry_id):
    entry = CompanyData.query.get_or_404(entry_id)
    company_id = entry.company_id
    db.session.delete(entry)
    db.session.commit()
    return redirect(url_for('manage', company_id=company_id))

@app.route('/agent/<int:company_id>')
def agent(company_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('agent.html', company_id=company_id)

@app.route('/agent_api', methods=['POST'])
def agent_api():
    data = request.get_json()
    message = data.get('message', '')

    if not message:
        return jsonify({"answer": "No input provided.", "script": ""})

    # OpenAI response
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message}],
        max_tokens=500
    )

    answer = response.choices[0].message.content
    return jsonify({"answer": answer, "script": "Dynamic script goes here."})

if __name__ == '__main__':
    app.run(debug=True)
