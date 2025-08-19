from app import app, db, User
from werkzeug.security import generate_password_hash

# Replace these with your desired admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123"

with app.app_context():
    # Check if admin already exists
    if not User.query.filter_by(username=ADMIN_USERNAME).first():
        admin = User(
            username=ADMIN_USERNAME,
            password_hash=generate_password_hash(ADMIN_PASSWORD)
        )
        db.session.add(admin)
        db.session.commit()
        print(f"Admin user '{ADMIN_USERNAME}' created successfully!")
    else:
        print(f"Admin user '{ADMIN_USERNAME}' already exists.")
