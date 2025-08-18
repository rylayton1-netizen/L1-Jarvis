import os
import sqlite3
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, SelectField
from wtforms.validators import DataRequired
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import pandas as pd
from PyPDF2 import PdfReader

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)
secret_key = os.getenv('SECRET_KEY')
database_url = os.getenv('DATABASE_URL')

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Models for Multi-Tenancy
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)  # Use hashed passwords in production

class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    knowledge_db = db.Column(db.String(120), nullable=False)  # e.g., 'company1.db'

# Forms
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class CompanyForm(FlaskForm):
    name = StringField('Company Name', validators=[DataRequired()])
    submit = SubmitField('Create Company')

class UploadForm(FlaskForm):
    company_id = SelectField('Select Company', choices=[], coerce=int)
    file = FileField('Upload CSV/PDF')
    url = StringField('Crawl URL')
    submit = SubmitField('Submit')

# User Loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.password == form.password.data:  # Hash passwords in production
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    form = CompanyForm()
    if form.validate_on_submit():
        knowledge_db = f"{form.name.data.replace(' ', '_').lower()}.db"
        new_company = Company(name=form.name.data, knowledge_db=knowledge_db)
        db.session.add(new_company)
        db.session.commit()
        # Initialize company database
        conn = sqlite3.connect(knowledge_db)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS knowledge
                          (id INTEGER PRIMARY KEY, content TEXT)''')
        conn.commit()
        conn.close()
        flash('Company created')
    companies = Company.query.all()
    return render_template('dashboard.html', form=form, companies=companies)

@app.route('/manage/<int:company_id>', methods=['GET', 'POST'])
@login_required
def manage(company_id):
    company = Company.query.get_or_404(company_id)
    form = UploadForm()
    form.company_id.choices = [(c.id, c.name) for c in Company.query.all()]
    form.company_id.data = company_id
    if form.validate_on_submit():
        conn = sqlite3.connect(company.knowledge_db)
        cursor = conn.cursor()
        if form.file.data:
            file = form.file.data
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                content = df.to_string()
            elif file.filename.endswith('.pdf'):
                reader = PdfReader(file)
                content = ''.join(page.extract_text() + '\n' for page in reader.pages)
            cursor.execute("INSERT INTO knowledge (content) VALUES (?)", (content,))
        if form.url.data:
            response = requests.get(form.url.data)
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.get_text()
            cursor.execute("INSERT INTO knowledge (content) VALUES (?)", (content,))
        conn.commit()
        conn.close()
        flash('Data added')
    return render_template('manage.html', form=form, company=company)

@app.route('/agent/<company_name>')
def agent(company_name):
    company = Company.query.filter_by(name=company_name).first_or_404()
    return render_template('index.html')  # Your Grok-like UI

@app.route('/query/<company_name>', methods=['POST'])
def query(company_name):
    company = Company.query.filter_by(name=company_name).first_or_404()
    query = request.form['query']
    conn = sqlite3.connect(company.knowledge_db)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM knowledge WHERE content LIKE ?", ('%' + query + '%',))
    results = cursor.fetchall()
    context = ' '.join([row[0] for row in results])
    conn.close()
    prompt = f"Based on this information: {context}\n\nCurrent date: August 18, 2025. Answer the question for a phone support agent handling inquiries about Jase Medical emergency medications: {query}\nSuggest a professional script to say on the phone."
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return jsonify({'answer': answer})

@app.after_request
def add_headers(response):
    response.headers['X-Frame-Options'] = 'ALLOWALL'
    return response

with app.app_context():
    db.create_all()
    if not User.query.first():
        admin = User(username='admin', password='password')  # Change in production
        db.session.add(admin)
        db.session.commit()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))