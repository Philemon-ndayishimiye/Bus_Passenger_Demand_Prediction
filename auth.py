

import bcrypt
from datetime import timedelta
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity,
    get_jwt,
)

from database import db
from models import User, TokenBlocklist

# Blueprint 
auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# Token lifetime 
TOKEN_EXPIRY = timedelta(hours=3)


# REGISTER — POST /auth/register

@auth_bp.route("/register", methods=["POST"])
def register():
    """
    Create a new user account and return a 3-hour JWT token.

    Expected JSON body:
    {
        "name":     "John Doe",
        "email":    "john@example.com",
        "password": "secret123"
    }
    """
    data = request.get_json(force=True)

    #  Validate required fields 
    for field in ["name", "email", "password"]:
        if not data.get(field):
            return jsonify({"error": f"'{field}' is required"}), 400

    #  Check for duplicate email 
    if User.query.filter_by(email=data["email"].lower()).first():
        return jsonify({"error": "Email already registered"}), 409

    # Hash password (never store plain text) 
    hashed_pw = bcrypt.hashpw(
        data["password"].encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    # Save user 
    new_user = User(
        name     = data["name"].strip(),
        email    = data["email"].lower().strip(),
        password = hashed_pw,
    )
    db.session.add(new_user)
    db.session.commit()

    # Issue 3-hour token 
    token = create_access_token(
        identity=str(new_user.id),
        expires_delta=TOKEN_EXPIRY
    )

    return jsonify({
        "message": "Account created successfully",
        "token":   token,
        "user":    new_user.to_dict(),
    }), 201


# LOGIN — POST /auth/login
@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Log in with email + password. Returns a 3-hour JWT token.

    Expected JSON body:
    {
        "email":    "john@example.com",
        "password": "secret123"
    }
    """
    data = request.get_json(force=True)

    # Validate fields 
    if not data.get("email") or not data.get("password"):
        return jsonify({"error": "Email and password are required"}), 400

    #  Look up user
    user = User.query.filter_by(email=data["email"].lower().strip()).first()

    # Verify password
    if not user or not bcrypt.checkpw(
        data["password"].encode("utf-8"),
        user.password.encode("utf-8"),
    ):
        return jsonify({"error": "Invalid email or password"}), 401

    # Issue 3-hour token
    token = create_access_token(
        identity=str(user.id),
        expires_delta=TOKEN_EXPIRY
    )

    return jsonify({
        "message": "Login successful",
        "token":   token,
        "user":    user.to_dict(),
    }), 200


# ME — GET /auth/me  
@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    """
    Returns the currently logged-in user's profile.
    Requires:  Authorization: Bearer <token>
    """
    user_id = int(get_jwt_identity())
    user    = User.query.get(user_id)

    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"user": user.to_dict()}), 200

# LOGOUT — POST /auth/logout 
@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    """
    Revokes the current token by saving its unique ID (jti) to the blocklist.
    Any future request using this token will receive 401 Unauthorized.
    Requires:  Authorization: Bearer <token>
    """
    jti = get_jwt()["jti"]                    
    db.session.add(TokenBlocklist(jti=jti))
    db.session.commit()

    return jsonify({"message": "Logged out successfully"}), 200