
import os
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Create the shared db instance 
# This object is imported everywhere one instance for the whole app
db = SQLAlchemy()


def init_db(app):
    """
    Call this in api.py to attach the database to the Flask app.
    Also creates all tables if they don't exist yet.
    """
    # Read the database URL from .env
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is not set in your .env file!")

    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False 

    # Attach SQLAlchemy to the app
    db.init_app(app)

    # Create tables automatically if they don't exist
    with app.app_context():
        from models import User, Prediction  
        db.create_all()
        print(" Database tables created ")
