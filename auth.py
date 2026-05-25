from flask import Blueprint, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from database import SessionLocal, User
from extensions import bcrypt

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/auth/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        db = SessionLocal()
        user = db.query(User).filter_by(username=username).first()
        if user and user.password_hash and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user, remember=True)
            db.close()
            return redirect(url_for("index"))
        db.close()
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@auth_bp.route("/auth/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip() or None
        password = request.form.get("password", "")
        if not username or not password:
            error = "Username and password are required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        else:
            db = SessionLocal()
            existing = db.query(User).filter_by(username=username).first()
            if existing:
                db.close()
                error = "Username already taken."
            else:
                user = User(
                    username=username,
                    email=email,
                    password_hash=bcrypt.generate_password_hash(password).decode("utf-8"),
                    role="researcher",
                )
                db.add(user)
                db.commit()
                login_user(user, remember=True)
                db.close()
                return redirect(url_for("index"))
    return render_template("register.html", error=error)


@auth_bp.route("/auth/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
