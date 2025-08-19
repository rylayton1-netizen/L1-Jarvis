import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import openai
from dotenv import load_dotenv
from sqlalchemy import text

# Load environment variables
load_dotenv()

# ========================
# CONFIGURATION
# ========================
def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey")

    # Database configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
        "DATABASE_URL", "sqlite:///l1_jarvis.db"
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Upload folder (cloud-friendly)
    UPLOAD_FOLDER = os.environ.get('UPLOAD_PATH', 'Uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # Allowed file extensions
    app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'csv', 'docx'}

    return app

app = create_app()
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# ========================
# DATABASE MODELS
# ========================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)

class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)

class CompanyData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=False)
    name_or_url = db.Column(db.String(255))
    filename = db.Column(db.String(255))

# ========================
# HELPER FUNCTIONS
# ========================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def check_database():
    try:
        db.session.execute(text('SELECT 1'))
    except Exception as e:
        app.logger.error(f"Database connection failed: {e}")
        raise RuntimeError("Cannot connect to the database. Check DATABASE_URL.")

def create_default_admin():
    if not User.query.filter_by(username="admin").first():
        admin = User(username="admin", password_hash=generate_password_hash("admin123"))
        db.session.add(admin)
        db.session.commit()
        print("Default admin created: username='admin', password='admin123'")

# ========================
# ROUTES
# ========================
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        flash("Invalid username or password", "error")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        company_name = request.form.get('company_name')
        if company_name:
            new_company = Company(name=company_name)
            db.session.add(new_company)
            db.session.commit()
            flash("Company created", "success")
            return redirect(url_for('dashboard'))
    companies = Company.query.all()
    return render_template('dashboard.html', companies=companies)

@app.route('/delete_company/<int:company_id>')
def delete_company(company_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    company = Company.query.get_or_404(company_id)
    CompanyData.query.filter_by(company_id=company.id).delete()
    db.session.delete(company)
    db.session.commit()
    flash("Company deleted", "success")
    return redirect(url_for('dashboard'))

@app.route('/manage/<int:company_id>', methods=['GET', 'POST'])
def manage(company_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    company = Company.query.get_or_404(company_id)
    if request.method == 'POST':
        file = request.files.get('file')
        url_text = request.form.get('url')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            entry = CompanyData(company_id=company.id, filename=filename, name_or_url=filename)
            db.session.add(entry)
        if url_text:
            entry = CompanyData(company_id=company.id, name_or_url=url_text)
            db.session.add(entry)
        db.session.commit()
        flash("Data submitted", "success")
        return redirect(url_for('manage', company_id=company.id))
    data = CompanyData.query.filter_by(company_id=company.id).all()
    return render_template('manage.html', company=company, data=data)

@app.route('/delete_entry/<int:entry_id>')
def delete_entry(entry_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    entry = CompanyData.query.get_or_404(entry_id)
    if entry.filename:
        try:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], entry.filename))
        except FileNotFoundError:
            pass
    db.session.delete(entry)
    db.session.commit()
    flash("Entry deleted", "success")
    return redirect(request.referrer or url_for('dashboard'))

@app.route('/agent/<int:company_id>')
def agent(company_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    company = Company.query.get_or_404(company_id)
    return render_template('agent.html', company=company)

@app.route('/agent_api', methods=['POST'])
def agent_api():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    user_message = data.get('message')
    if not user_message:
        return jsonify({"answer": "No input received", "script": ""})
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI call center assistant."},
            {"role": "user", "content": user_message}
        ],
        max_tokens=500
    )
    answer_text = response['choices'][0]['message']['content']
    answer, script = answer_text, ""
    return jsonify({"answer": answer, "script": script})

# ========================
# INITIALIZE DB & ADMIN
# ========================
with app.app_context():
    check_database()
    db.create_all()
    create_default_admin()

# ========================
# RUN APP
# ========================
if __name__ == '__main__':
    app.run(debug=True)
