import os
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
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
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not set in .env")
secret_key = os.getenv('SECRET_KEY')
if not secret_key:
    raise ValueError("SECRET_KEY not set in .env")
database_url = os.getenv('DATABASE_URL')
if not database_url:
    raise ValueError("DATABASE_URL not set in .env")

try:
    client = OpenAI(api_key=openai_api_key)
except Exception as e:
    raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class Company(db.Model):
    __tablename__ = 'companies'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    schema_name = db.Column(db.String(120), unique=True, nullable=False)

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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def execute_in_schema(schema_name, query, params=None):
    is_sqlite = database_url.startswith('sqlite')
    with db.engine.connect() as conn:
        if not is_sqlite:
            conn.execute(f"SET search_path TO {schema_name}")
        if params:
            conn.execute(query, params)
        else:
            conn.execute(query)
        conn.commit()

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.password == form.password.data:
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
        schema_name = form.name.data.replace(' ', '_').lower()
        new_company = Company(name=form.name.data, schema_name=schema_name)
        db.session.add(new_company)
        db.session.commit()
        is_sqlite = database_url.startswith('sqlite')
        with db.engine.connect() as conn:
            if not is_sqlite:
                conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema_name if not is_sqlite else 'public'}.knowledge (
                    id SERIAL PRIMARY KEY,
                    content TEXT
                )
            """)
            conn.commit()
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
        is_sqlite = database_url.startswith('sqlite')
        with db.engine.connect() as conn:
            if not is_sqlite:
                conn.execute(f"SET search_path TO {company.schema_name}")
            schema = company.schema_name if not is_sqlite else 'public'
            if form.file.data:
                file = form.file.data
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file)
                    content = df.to_string()
                elif file.filename.endswith('.pdf'):
                    reader = PdfReader(file)
                    content = ''.join(page.extract_text() + '\n' for page in reader.pages)
                conn.execute(f"INSERT INTO {schema}.knowledge (content) VALUES (%s)", (content,))
            if form.url.data:
                response = requests.get(form.url.data)
                soup = BeautifulSoup(response.text, 'html.parser')
                content = soup.get_text()
                conn.execute(f"INSERT INTO {schema}.knowledge (content) VALUES (%s)", (content,))
            conn.commit()
        flash('Data added')
    return render_template('manage.html', form=form, company=company)

@app.route('/agent/<company_name>')
def agent(company_name):
    Company.query.filter_by(name=company_name).first_or_404()
    return render_template('index.html')

@app.route('/query/<company_name>', methods=['POST'])
def query(company_name):
    company = Company.query.filter_by(name=company_name).first_or_404()
    query = request.form['query']
    is_sqlite = database_url.startswith('sqlite')
    with db.engine.connect() as conn:
        if not is_sqlite:
            conn.execute(f"SET search_path TO {company.schema_name}")
        schema = company.schema_name if not is_sqlite else 'public'
        result = conn.execute(f"SELECT content FROM {schema}.knowledge WHERE content LIKE %s", ('%' + query + '%',))
        results = result.fetchall()
        context = ' '.join([row[0] for row in results])
    prompt = f"Based on this information: {context}\n\nCurrent date: August 18, 2025. Answer the question for a phone support agent handling inquiries about {company_name} emergency medications: {query}\nThen, provide a professional script to say on the phone, prefixed with 'Script:'."
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
    except Exception as e:
        return jsonify({'answer': f"Error calling OpenAI: {str(e)}"})
    return jsonify({'answer': answer})

@app.after_request
def add_headers(response):
    response.headers['X-Frame-Options'] = 'ALLOWALL'
    return response

with app.app_context():
    try:
        db.create_all()
        if not User.query.first():
            admin = User(username='admin', password='password')
            db.session.add(admin)
            db.session.commit()
    except Exception as e:
        print(f"Database initialization error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)