from api import app
from database import db
from sqlalchemy import text

with app.app_context():
    with db.engine.connect() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) NOT NULL DEFAULT 'user';"))
        conn.commit()
    print("role column added successfully!")

with app.app_context():
    with db.engine.connect() as conn:
        conn.execute(text("""
            UPDATE users 
            SET role = 'admin' 
            WHERE email = 'celineaba@gmail.com';
        """))
        conn.commit()
    print("User role updated to admin successfully")