

import os
import bcrypt
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

# Import the Flask app 
from api import app
from database import db
from models import User


def seed_default_user():
    """Create the default admin user if they don't exist yet."""

    with app.app_context():

        email    = os.getenv("DEFAULT_USER_EMAIL",    "admin@bilstm.com")
        password = os.getenv("DEFAULT_USER_PASSWORD", "Admin@1234")
        name     = os.getenv("DEFAULT_USER_NAME",     "Admin")

        # Check if user already exists
        existing = User.query.filter_by(email=email.lower()).first()
        if existing:
            print(f"Default user already exists: {email}")
            return

        # Hash the password 
        hashed_pw = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        ).decode("utf-8")

        #Insert into database 
        admin = User(
            name     = name,
            email    = email.lower(),
            password = hashed_pw,
            role     = "admin",
        )
        db.session.add(admin)
        db.session.commit()

        print(" Default admin user created!")
        print(f"   Email:    {email}")
        print(f"   Password: {password}")
        print("    Change this password after first login!")


if __name__ == "__main__":
    seed_default_user()
