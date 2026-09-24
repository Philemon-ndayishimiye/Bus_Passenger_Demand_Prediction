
import bcrypt
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy import func
from datetime import datetime, timedelta

from database import db
from models import User

# Blueprint
users_bp = Blueprint("users", __name__, url_prefix="/users")


# Helper: check caller is admin
def require_admin():
    """
    Returns (user, None) if the caller is an admin,
    or (None, error_response) if not.
    """
    user_id = int(get_jwt_identity())
    user    = User.query.get(user_id)
    if not user:
        return None, (jsonify({"error": "User not found"}), 404)
    if user.role != "admin":
        return None, (jsonify({"error": "Admin access required"}), 403)
    return user, None

# LIST ALL USERS — GET /users

@users_bp.route("", methods=["GET"])
@jwt_required()
def list_users():
    """
    Returns all users with their prediction count.
    Requires admin role.
    """
    _, err = require_admin()
    if err:
        return err

    users = User.query.order_by(User.created_at.desc()).all()
    return jsonify({
        "total": len(users),
        "users": [u.to_dict_with_predictions() for u in users],
    }), 200

# CREATE USER — POST /users
@users_bp.route("", methods=["POST"])
@jwt_required()
def create_user():
    """
    Admin creates a new user account.

    Expected JSON body:
    {
        "name":     "Jane Doe",
        "email":    "jane@example.com",
        "password": "secret123",
        "role":     "user"        ← optional, defaults to "user"
    }
    """
    _, err = require_admin()
    if err:
        return err

    data = request.get_json(force=True)

    for field in ["name", "email", "password"]:
        if not data.get(field):
            return jsonify({"error": f"'{field}' is required"}), 400

    if User.query.filter_by(email=data["email"].lower()).first():
        return jsonify({"error": "Email already registered"}), 409

    hashed_pw = bcrypt.hashpw(
        data["password"].encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    role = data.get("role", "user")
    if role not in ("admin", "user"):
        role = "user"

    new_user = User(
        name     = data["name"].strip(),
        email    = data["email"].lower().strip(),
        password = hashed_pw,
        role     = role,
    )
    db.session.add(new_user)
    db.session.commit()

    return jsonify({
        "message": "User created successfully",
        "user":    new_user.to_dict_with_predictions(),
    }), 201

# DELETE USER — DELETE /users/<id>
@users_bp.route("/<int:user_id>", methods=["DELETE"])
@jwt_required()
def delete_user(user_id: int):
    """
    Admin deletes a user by ID.
    An admin cannot delete their own account.
    """
    caller, err = require_admin()
    if err:
        return err

    if caller.id == user_id:
        return jsonify({"error": "You cannot delete your own account"}), 400

    target = User.query.get(user_id)
    if not target:
        return jsonify({"error": "User not found"}), 404

    db.session.delete(target)
    db.session.commit()

    return jsonify({"message": f"User '{target.name}' deleted successfully"}), 200

# STATS — GET /users/stats
@users_bp.route("/stats", methods=["GET"])
@jwt_required()
def user_stats():
    """
    Returns summary user stats for the dashboard.
    Any authenticated user can call this (not admin-only).
    """
    total_users   = User.query.count()
    admins        = User.query.filter_by(role="admin").count()
    regular_users = total_users - admins

    one_week_ago  = datetime.utcnow() - timedelta(days=7)
    new_this_week = User.query.filter(User.created_at >= one_week_ago).count()

    return jsonify({
        "total_users":   total_users,
        "new_this_week": new_this_week,
        "admins":        admins,
        "regular_users": regular_users,
    }), 200
