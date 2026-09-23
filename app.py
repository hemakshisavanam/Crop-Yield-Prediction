"""
AgriSense - AI-Based Crop Yield Prediction and Smart Farming Assistant
Flask Web Application Backend
"""

import os
import sqlite3
import datetime
import logging
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

from utils.crops_data import (
    get_metadata, get_all_crops, get_all_areas, search_crops,
    get_crop_details, get_dashboard_chart_data,
    get_crop_categories, get_all_selectable_crops, resolve_crop,
    EXTENDED_CROPS_CATALOG
)
from utils.assistant import generate_farming_advice, get_response

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("agrisense")

# Load environment configuration
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "agrisense_production_secret_key_2026_d912")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "users.db")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ===============================
# Model Loading & Auto-Recovery Pipeline
# ===============================
model = None
area_encoder = None
crop_encoder = None
metadata = get_metadata()

def get_model_paths():
    model_path = os.path.join(MODEL_DIR, "crop_yield_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "crop_model.pkl")
    area_encoder_path = os.path.join(MODEL_DIR, "area_encoder.pkl")
    crop_encoder_path = os.path.join(MODEL_DIR, "crop_encoder.pkl")
    return model_path, area_encoder_path, crop_encoder_path

def load_or_train_models(force_train=False):
    """
    Safely loads ML model and encoders from disk.
    If missing (e.g. on fresh cloud deployments), automatically triggers the
    training pipeline and loads newly generated artifacts.
    """
    global model, area_encoder, crop_encoder, metadata
    model_path, area_encoder_path, crop_encoder_path = get_model_paths()

    missing_on_disk = (
        force_train or
        not os.path.exists(model_path) or
        not os.path.exists(area_encoder_path) or
        not os.path.exists(crop_encoder_path)
    )

    if missing_on_disk:
        logger.warning("One or more ML model artifacts not found on disk. Initiating automated training pipeline...")
        try:
            from model.train_model import train
            m, le_a, le_c, meta = train()
            model = m
            area_encoder = le_a
            crop_encoder = le_c
            metadata = meta
            logger.info("Automated training pipeline completed successfully. Model loaded into memory.")
            return model, area_encoder, crop_encoder
        except Exception as e:
            logger.error(f"Failed to auto-train model artifacts: {e}", exc_info=True)

    try:
        if os.path.exists(model_path) and model is None:
            model = joblib.load(model_path)
            logger.info(f"Loaded crop yield model from: {model_path}")
        if os.path.exists(area_encoder_path) and area_encoder is None:
            area_encoder = joblib.load(area_encoder_path)
            logger.info(f"Loaded area encoder ({len(area_encoder.classes_)} regions)")
        if os.path.exists(crop_encoder_path) and crop_encoder is None:
            crop_encoder = joblib.load(crop_encoder_path)
            logger.info(f"Loaded crop encoder ({len(crop_encoder.classes_)} crops)")
    except Exception as e:
        logger.error(f"Error loading model artifacts from disk: {e}", exc_info=True)

    return model, area_encoder, crop_encoder

def ensure_models():
    """
    Verifies that model and encoders are ready for inference.
    Attempts auto-recovery if any are None.
    """
    global model, area_encoder, crop_encoder
    if model is None or area_encoder is None or crop_encoder is None:
        load_or_train_models()
    return (model is not None and area_encoder is not None and crop_encoder is not None)

# Initialize models at application startup
load_or_train_models()
metadata = get_metadata()

# ===============================
# Database Initialization & Migration
# ===============================

def get_db_connection():
    conn = sqlite3.connect(DATABASE, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    # Enable Write-Ahead Logging (WAL) and busy timeout for high-concurrency resilience (e.g. OneDrive)
    try:
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
    except Exception:
        pass

    # Users Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Predictions Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        area TEXT NOT NULL,
        crop TEXT NOT NULL,
        year INTEGER NOT NULL,
        rainfall REAL NOT NULL,
        pesticides REAL NOT NULL,
        temperature REAL NOT NULL,
        prediction REAL NOT NULL,
        yield_kg_ha REAL,
        yield_tonnes_ha REAL,
        prediction_type TEXT DEFAULT 'Standard Prediction',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Check for schema migrations on existing database
    cur.execute("PRAGMA table_info(users)")
    user_cols = [row["name"] for row in cur.fetchall()]
    if "created_at" not in user_cols:
        cur.execute("ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT NULL")
        cur.execute("UPDATE users SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")

    cur.execute("PRAGMA table_info(predictions)")
    cols = [row["name"] for row in cur.fetchall()]
    if "yield_kg_ha" not in cols:
        cur.execute("ALTER TABLE predictions ADD COLUMN yield_kg_ha REAL")
    if "yield_tonnes_ha" not in cols:
        cur.execute("ALTER TABLE predictions ADD COLUMN yield_tonnes_ha REAL")
    if "prediction_type" not in cols:
        cur.execute("ALTER TABLE predictions ADD COLUMN prediction_type TEXT DEFAULT 'Standard Prediction'")
    if "created_at" not in cols:
        cur.execute("ALTER TABLE predictions ADD COLUMN created_at TIMESTAMP DEFAULT NULL")
        cur.execute("UPDATE predictions SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")

    conn.commit()
    conn.close()

# Initialize tables immediately
init_db()

# Global Context Processor for Navigation and Metadata
@app.context_processor
def inject_global_vars():
    user = session.get("user", None)
    return {
        "current_user": user,
        "app_name": "AgriSense",
        "app_tagline": "AI-Based Crop Yield Prediction and Smart Farming Assistant",
        "supported_crops_count": metadata.get("crops_count", 10),
        "supported_areas_count": metadata.get("areas_count", 101),
        "training_year_min": metadata.get("ranges", {}).get("year", {}).get("min", 1990),
        "training_year_max": metadata.get("ranges", {}).get("year", {}).get("max", 2013)
    }

# Template Filters
@app.template_filter('format')
@app.template_filter('number_format')
def format_number(val, default='0'):
    try:
        if val is None or val == "":
            return default
        f = float(val)
        if f.is_integer():
            return f"{int(f):,}"
        return f"{f:,.2f}"
    except (ValueError, TypeError):
        return str(val)

@app.template_filter('markdown')
def render_markdown(text):
    import re
    from markupsafe import Markup
    if not text:
        return ""
    # Header level 3
    text = re.sub(r'^###\s+(.+)$', r'<h5 class="advice-heading mt-3 mb-2"><i class="fa-solid fa-seedling text-success me-2"></i>\1</h5>', text, flags=re.MULTILINE)
    # Header level 2
    text = re.sub(r'^##\s+(.+)$', r'<h4 class="fw-bold text-success mt-3 mb-2">\1</h4>', text, flags=re.MULTILINE)
    # Bold **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italics *text*
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Bullet points
    text = re.sub(r'^\s*[\*\-]\s+(.+)$', r'<li class="mb-1">\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'((?:<li class="mb-1">.*?</li>\s*)+)', r'<ul class="ps-3 mb-2">\1</ul>', text, flags=re.DOTALL)
    # Newlines
    text = text.replace('\n\n', '<br>')
    return Markup(text)


# ===============================
# Authentication Routes
# ===============================

@app.route("/register", methods=["GET", "POST"])
@app.route("/signup", methods=["GET", "POST"])
def signup():
    # If user is already logged in, redirect them directly to the dashboard
    if "user" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("All fields are required.", "warning")
            return render_template("signup.html", username=username, email=email)

        if len(password) < 4:
            flash("Password must be at least 4 characters.", "warning")
            return render_template("signup.html", username=username, email=email)

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            hashed_pwd = generate_password_hash(password)
            cur.execute(
                "INSERT INTO users(username, email, password) VALUES (?, ?, ?)",
                (username, email, hashed_pwd)
            )
            conn.commit()
            user_id = cur.lastrowid

            # Automatically log in the user so clicking 'Get Started' registers and enters the app immediately
            session["user"] = username
            session["user_id"] = user_id

            flash(f"Registration successful! Welcome to AgriSense, {username}.", "success")
            next_url = request.args.get("next")
            return redirect(next_url or url_for("dashboard"))
        except sqlite3.IntegrityError:
            flash("An account with this email address already exists. Please log in instead or use another email.", "danger")
            return render_template("signup.html", username=username, email=email, email_exists=True)
        except Exception as e:
            flash(f"An unexpected error occurred during registration: {e}", "danger")
            return render_template("signup.html", username=username, email=email)
        finally:
            conn.close()

    return render_template("signup.html", username="", email="")

@app.route("/login", methods=["GET", "POST"])
def login():
    # If user is already logged in, redirect them directly to the dashboard on GET requests
    if request.method == "GET" and "user" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cur.fetchone()
        conn.close()

        if user:
            # Verify hashed password or allow legacy plaintext if not yet updated
            pwd_matches = False
            user_pwd = user["password"]
            if user_pwd.startswith("scrypt:") or user_pwd.startswith("pbkdf2:"):
                pwd_matches = check_password_hash(user_pwd, password)
            else:
                pwd_matches = (user_pwd == password)

            if pwd_matches:
                session["user"] = user["username"]
                session["user_id"] = user["id"]
                flash(f"Welcome back, {user['username']}!", "success")
                next_url = request.args.get("next")
                return redirect(next_url or url_for("dashboard"))

        flash("Invalid email or password. Please verify and try again.", "danger")
        return render_template("login.html", email=email)

    return render_template("login.html", email=request.args.get("email", ""))

@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("user_id", None)
    flash("You have been securely logged out.", "info")
    return redirect(url_for("home"))

# ===============================
# Core Page Routes
# ===============================

@app.route("/")
def home():
    areas = get_all_areas()
    crops = get_all_crops()
    metrics = metadata.get("metrics", {})
    return render_template(
        "home.html",
        areas=areas,
        crops=crops,
        metrics=metrics,
        metadata=metadata
    )

@app.route("/about")
def about():
    return render_template("about.html", metadata=metadata)

# ===============================
# Yield Prediction
# ===============================

@app.route("/predict")
def predict():
    areas = get_all_areas()
    crops = get_all_crops()
    all_selectable_crops = get_all_selectable_crops()
    crop_categories = get_crop_categories()
    ranges = metadata.get("ranges", {})

    # Pre-fill parameters if arriving from "Explore Another Crop" or preset links
    prefill = {
        "area": request.args.get("area", "India"),
        "crop": request.args.get("crop", "Maize"),
        "year": request.args.get("year", "2026"),
        "rainfall": request.args.get("rainfall", "1000"),
        "temperature": request.args.get("temperature", "24"),
        "pesticides": request.args.get("pesticides", "15000")
    }

    return render_template(
        "predict.html",
        areas=areas,
        crops=crops,
        all_selectable_crops=all_selectable_crops,
        crop_categories=crop_categories,
        ranges=ranges,
        prefill=prefill
    )

@app.route("/prediction", methods=["POST"])
def prediction():
    if not ensure_models():
        flash("The predictive machine learning model is currently initializing. Please try again in a moment.", "warning")
        return redirect(url_for("predict"))

    area_name = request.form.get("area", "").strip()
    crop_name = request.form.get("crop", "").strip()
    manual_category = request.form.get("manual_category", "").strip()
    is_manual_mode = request.form.get("is_manual_mode", "0") == "1"

    if not crop_name:
        flash("Please select a crop variety or enter a crop name manually.", "warning")
        return redirect(url_for("predict"))

    try:
        year = int(request.form.get("year", 2026))
        rainfall = float(request.form.get("rainfall", 1000))
        pesticides = float(request.form.get("pesticides", 10000))
        temperature = float(request.form.get("temperature", 25))
    except ValueError:
        flash("Please enter valid numerical values for year, rainfall, pesticides, and temperature.", "danger")
        return redirect(url_for("predict"))

    # Scientific validation
    if rainfall < 0 or rainfall > 6000:
        flash("Average annual rainfall must be between 0 and 6,000 mm/year.", "danger")
        return redirect(url_for("predict"))

    if temperature < -20 or temperature > 60:
        flash("Average temperature must be within a realistic range (-20°C to 60°C).", "danger")
        return redirect(url_for("predict"))

    if pesticides < 0:
        flash("Pesticide usage cannot be negative.", "danger")
        return redirect(url_for("predict"))

    try:
        # Validate geographic area against trained encoders
        if not area_name or area_name not in area_encoder.classes_:
            flash(f"Country '{area_name}' is not in the training dataset. Please select an available country.", "warning")
            return redirect(url_for("predict"))

        # Resolve crop (Core 10 FAO, Extended Catalog, or Manual Input)
        resolution = resolve_crop(crop_name, manual_category=manual_category)
        resolved_crop_name = resolution["crop"]
        archetype_crop = resolution["archetype"]
        scaling_factor = resolution["scaling_factor"]
        crop_category = resolution["category"]
        is_manual = resolution["is_manual"] or is_manual_mode
        is_extended = resolution["is_extended"]
        is_core = resolution["is_core"]

        if archetype_crop not in crop_encoder.classes_:
            archetype_crop = "Maize"

        # Encode categories
        encoded_area = area_encoder.transform([area_name])[0]
        encoded_crop = crop_encoder.transform([archetype_crop])[0]

        input_df = pd.DataFrame([[
            encoded_area,
            encoded_crop,
            year,
            rainfall,
            pesticides,
            temperature
        ]], columns=[
            "Area",
            "Item",
            "Year",
            "average_rain_fall_mm_per_year",
            "pesticides_tonnes",
            "avg_temp"
        ])

        # Model prediction on archetype
        raw_predicted_hg_ha = max(0.0, float(model.predict(input_df)[0]))
        # Apply calibrated scaling factor for extended / manual crop
        predicted_yield_hg_ha = max(0.0, round(raw_predicted_hg_ha * scaling_factor, 1))

        # Mathematically exact conversions
        # 1 hg/ha = 0.1 kg/ha
        yield_kg_ha = round(predicted_yield_hg_ha * 0.1, 2)
        # 1 tonne = 1000 kg => tonnes/ha = kg/ha / 1000
        yield_tonnes_ha = round(yield_kg_ha / 1000.0, 3)

        # Determine Prediction Type
        training_max_year = metadata.get("ranges", {}).get("year", {}).get("max", 2013)
        training_min_year = metadata.get("ranges", {}).get("year", {}).get("min", 1990)
        is_future = (year > training_max_year)

        if is_manual:
            prediction_type = "Manual Crop Estimate"
        elif is_extended:
            prediction_type = "Extended Agronomic Estimate"
        elif is_future:
            prediction_type = "Future-Year Model Estimate"
        elif year < training_min_year:
            prediction_type = "Historical Extrapolation"
        else:
            prediction_type = "Within Historical Range"

        # Out-of-bounds checks against observed distributions
        ranges = metadata.get("ranges", {})
        warnings = []
        if rainfall < ranges.get("rainfall_mm", {}).get("min", 50) or rainfall > ranges.get("rainfall_mm", {}).get("max", 3500):
            warnings.append(f"Rainfall ({rainfall} mm) is outside the typical training range ({ranges.get('rainfall_mm', {}).get('min')}–{ranges.get('rainfall_mm', {}).get('max')} mm).")
        if temperature < ranges.get("avg_temp_c", {}).get("min", 1.0) or temperature > ranges.get("avg_temp_c", {}).get("max", 31.0):
            warnings.append(f"Temperature ({temperature}°C) is outside observed training bounds ({ranges.get('avg_temp_c', {}).get('min')}–{ranges.get('avg_temp_c', {}).get('max')}°C).")
        if pesticides > ranges.get("pesticides_tonnes", {}).get("max", 370000):
            warnings.append(f"Pesticides ({pesticides} tonnes) exceeds maximum recorded training usage.")

        # Historical average comparison
        hist_avg_kg_ha = resolution.get("mean_yield_kg_ha", yield_kg_ha)
        hist_avg_hg_ha = round(hist_avg_kg_ha * 10.0, 1)

        # Save to history if user is logged in
        user = session.get("user")
        if user:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO predictions(
                        username, area, crop, year, rainfall, pesticides, temperature,
                        prediction, yield_kg_ha, yield_tonnes_ha, prediction_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user, area_name, resolved_crop_name, year, rainfall, pesticides, temperature,
                    predicted_yield_hg_ha, yield_kg_ha, yield_tonnes_ha, prediction_type
                ))
                conn.commit()
                conn.close()
            except Exception as dbe:
                logger.error(f"Failed to record prediction in history: {dbe}", exc_info=True)

        return render_template(
            "result.html",
            crop=resolved_crop_name,
            area=area_name,
            year=year,
            rainfall=rainfall,
            temperature=temperature,
            pesticides=pesticides,
            predicted_hg_ha=predicted_yield_hg_ha,
            yield_kg_ha=yield_kg_ha,
            yield_tonnes_ha=yield_tonnes_ha,
            prediction_type=prediction_type,
            is_future=is_future,
            is_manual=is_manual,
            is_extended=is_extended,
            is_core=is_core,
            archetype_crop=archetype_crop,
            scaling_factor=scaling_factor,
            crop_category=crop_category,
            source_label=resolution.get("source_label", "Agronomic Estimate"),
            training_min_year=training_min_year,
            training_max_year=training_max_year,
            warnings=warnings,
            crop_stats=resolution,
            hist_avg_kg_ha=hist_avg_kg_ha
        )
    except Exception as e:
        logger.exception("Prediction failed unexpectedly: %s", e)
        flash(f"An error occurred while calculating the yield prediction ({e}). Please try again.", "danger")
        return redirect(url_for("predict"))

# ===============================
# Crop Explorer Routes
# ===============================

@app.route("/crops")
def crops():
    all_crops = get_all_crops()
    crops_info = []
    for c in all_crops:
        details = get_crop_details(c)
        if details:
            crops_info.append(details)
    return render_template("crops.html", crops_info=crops_info)

@app.route("/crops/<crop_name>")
def crop_detail(crop_name):
    details = get_crop_details(crop_name)
    if not details:
        flash(f"Crop '{crop_name}' not found in the agricultural dataset.", "warning")
        return redirect(url_for("crops"))
    return render_template("crop_detail.html", details=details)

# ===============================
# Crop Comparison Routes
# ===============================

@app.route("/compare", methods=["GET", "POST"])
def compare():
    areas = get_all_areas()
    all_crops = get_all_crops()

    if request.method == "POST":
        area = request.form.get("area", "India")
        try:
            year = int(request.form.get("year", 2026))
            rainfall = float(request.form.get("rainfall", 1000))
            temperature = float(request.form.get("temperature", 24))
            pesticides = float(request.form.get("pesticides", 15000))
        except ValueError:
            flash("Invalid numerical values provided.", "danger")
            return redirect(url_for("compare"))

        selected_crops = request.form.getlist("selected_crops")
        if not selected_crops:
            # Default to top 3 crops if none checked
            selected_crops = ["Maize", "Rice, paddy", "Wheat"]

        if not ensure_models():
            flash("The comparison model is currently initializing. Please try again in a few moments.", "warning")
            return redirect(url_for("compare"))

        comparison_results = []
        try:
            if area_encoder and area in area_encoder.classes_:
                encoded_area = area_encoder.transform([area])[0]

                for c in selected_crops:
                    res = resolve_crop(c)
                    arch = res["archetype"]
                    factor = res["scaling_factor"]
                    if crop_encoder and arch in crop_encoder.classes_:
                        encoded_crop = crop_encoder.transform([arch])[0]
                        input_df = pd.DataFrame([[
                            encoded_area, encoded_crop, year, rainfall, pesticides, temperature
                        ]], columns=[
                            "Area", "Item", "Year", "average_rain_fall_mm_per_year",
                            "pesticides_tonnes", "avg_temp"
                        ])
                        pred_hg_ha = max(0.0, float(model.predict(input_df)[0])) * factor
                        pred_kg_ha = round(pred_hg_ha * 0.1, 1)
                        pred_tonnes_ha = round(pred_kg_ha / 1000.0, 3)

                        hist_avg_kg = res.get("mean_yield_kg_ha", 0.0)

                        comparison_results.append({
                            "crop": res["crop"],
                            "category": res["category"],
                            "yield_hg_ha": round(pred_hg_ha, 1),
                            "yield_kg_ha": pred_kg_ha,
                            "yield_tonnes_ha": pred_tonnes_ha,
                            "historical_avg_kg_ha": hist_avg_kg,
                            "diff_vs_historical": round(pred_kg_ha - hist_avg_kg, 1) if hist_avg_kg else 0.0
                        })

            # Sort by predicted yield descending
            comparison_results.sort(key=lambda x: x["yield_kg_ha"], reverse=True)
        except Exception as e:
            logger.exception("Error calculating crop comparisons: %s", e)
            flash("An error occurred while calculating crop comparisons. Please try again.", "danger")
            return redirect(url_for("compare"))

        return render_template(
            "compare.html",
            areas=areas,
            all_crops=all_crops,
            selected_crops=selected_crops,
            area=area,
            year=year,
            rainfall=rainfall,
            temperature=temperature,
            pesticides=pesticides,
            results=comparison_results
        )

    # GET default state
    default_crops = ["Maize", "Rice, paddy", "Wheat"]
    return render_template(
        "compare.html",
        areas=areas,
        all_crops=all_crops,
        selected_crops=default_crops,
        area="India",
        year=2026,
        rainfall=1000,
        temperature=24,
        pesticides=15000,
        results=None
    )

# ===============================
# Smart Farming Assistant Routes
# ===============================

@app.route("/assistant", methods=["GET", "POST"])
def assistant():
    areas = get_all_areas()
    crops = get_all_crops()
    all_selectable_crops = get_all_selectable_crops()
    advice_result = None
    chat_answer = None

    if request.method == "POST":
        action_type = request.form.get("action_type", "agronomy_calibration")

        # 1. Full Agronomy Calibrator (Country + Crop + Weather -> ML Model -> Gemini AI)
        if action_type == "agronomy_calibration":
            country = request.form.get("country", "India")
            crop = request.form.get("crop", "Maize")
            try:
                year = int(request.form.get("year", 2026))
                rainfall = float(request.form.get("rainfall", 1000))
                temperature = float(request.form.get("temperature", 24))
                pesticides = float(request.form.get("pesticides", 15000))
            except ValueError:
                flash("Please check your numbers.", "warning")
                return redirect(url_for("assistant"))

            # Calculate numerical yield via actual ML Model
            predicted_hg_ha = 0.0
            res = resolve_crop(crop)
            arch = res["archetype"]
            factor = res["scaling_factor"]
            ensure_models()
            if area_encoder and country in area_encoder.classes_ and crop_encoder and arch in crop_encoder.classes_ and model:
                enc_a = area_encoder.transform([country])[0]
                enc_c = crop_encoder.transform([arch])[0]
                inp = pd.DataFrame([[enc_a, enc_c, year, rainfall, pesticides, temperature]], columns=[
                    "Area", "Item", "Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"
                ])
                predicted_hg_ha = max(0.0, float(model.predict(inp)[0])) * factor

            hist_avg = res.get("mean_yield_kg_ha", 300.0) * 10.0

            # Generate Gemini or rule-based advice
            advice_result = generate_farming_advice(
                country=country,
                crop=crop,
                rainfall=rainfall,
                temperature=temperature,
                pesticides=pesticides,
                year=year,
                predicted_yield_hg_ha=predicted_hg_ha,
                historical_avg_hg_ha=hist_avg
            )

        # 2. General Agricultural Chat Q&A
        elif action_type == "chat_qa":
            question = request.form.get("question", "").strip()
            if question:
                chat_answer = get_response(question)

    return render_template(
        "assistant.html",
        areas=areas,
        crops=crops,
        advice_result=advice_result,
        chat_answer=chat_answer
    )

# ===============================
# Analytics Dashboard Routes
# ===============================

@app.route("/dashboard")
def dashboard():
    # User-specific statistics
    user = session.get("user")
    total_user_predictions = 0
    recent_predictions = []
    top_predicted_crop = "None"

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM predictions")
    platform_predictions_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM users")
    platform_users_count = cur.fetchone()[0]

    if user:
        cur.execute("SELECT COUNT(*) FROM predictions WHERE username = ?", (user,))
        total_user_predictions = cur.fetchone()[0]

        cur.execute("""
            SELECT crop, COUNT(*) as cnt FROM predictions
            WHERE username = ? GROUP BY crop ORDER BY cnt DESC LIMIT 1
        """, (user,))
        row = cur.fetchone()
        if row:
            top_predicted_crop = row["crop"]

        cur.execute("""
            SELECT * FROM predictions WHERE username = ?
            ORDER BY id DESC LIMIT 5
        """, (user,))
        recent_predictions = cur.fetchall()

    conn.close()

    metrics = metadata.get("metrics", {})
    ranges = metadata.get("ranges", {})

    return render_template(
        "dashboard.html",
        metadata=metadata,
        metrics=metrics,
        ranges=ranges,
        platform_predictions_count=platform_predictions_count,
        platform_users_count=platform_users_count,
        total_user_predictions=total_user_predictions,
        top_predicted_crop=top_predicted_crop,
        recent_predictions=recent_predictions
    )

# ===============================
# Prediction History Routes
# ===============================

@app.route("/history")
def history():
    if "user" not in session:
        flash("Please log in to view your prediction history.", "info")
        return redirect(url_for("login", next=request.url))

    user = session["user"]
    query = request.args.get("q", "").strip().lower()
    crop_filter = request.args.get("crop", "").strip()

    conn = get_db_connection()
    cur = conn.cursor()

    sql = "SELECT * FROM predictions WHERE username = ?"
    params = [user]

    if crop_filter:
        sql += " AND crop = ?"
        params.append(crop_filter)

    if query:
        sql += " AND (crop LIKE ? OR area LIKE ? OR prediction_type LIKE ?)"
        params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])

    sql += " ORDER BY id DESC"

    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    crops = get_all_crops()

    return render_template(
        "history.html",
        rows=rows,
        crops=crops,
        query=query,
        selected_crop=crop_filter,
        total_count=len(rows)
    )

@app.route("/delete/<int:id>", methods=["POST", "GET"])
def delete_prediction(id):
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions WHERE id = ? AND username = ?", (id, session["user"]))
    conn.commit()
    conn.close()

    flash("Prediction record removed from history.", "success")
    return redirect(url_for("history"))

# ===============================
# REST APIs for Dynamic UI
# ===============================

@app.route("/api/crops", methods=["GET"])
def api_crops():
    return jsonify({
        "success": True,
        "data": get_all_crops(),
        "selectable_crops": get_all_selectable_crops(),
        "categories": get_crop_categories(),
        "total": len(get_all_selectable_crops())
    })

@app.route("/api/crops/search", methods=["GET"])
def api_crop_search():
    query = request.args.get("q", "")
    res = search_crops(query)
    return jsonify({
        "success": True,
        "data": res
    })

@app.route("/api/crops/<crop_name>", methods=["GET"])
def api_crop_details(crop_name):
    details = get_crop_details(crop_name)
    if not details:
        return jsonify({"success": False, "error": f"Crop '{crop_name}' not found"}), 404
    return jsonify({
        "success": True,
        "data": details
    })

@app.route("/api/chart-data", methods=["GET"])
def api_chart_data():
    chart_data = get_dashboard_chart_data()
    return jsonify({
        "success": True,
        "data": chart_data
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not ensure_models():
        return jsonify({"success": False, "error": "Prediction model is currently initializing. Please retry in a few moments."}), 503

    data = request.get_json() or {}
    area = data.get("area")
    crop = data.get("crop")
    manual_category = data.get("manual_category")
    try:
        year = int(data.get("year", 2026))
        rainfall = float(data.get("rainfall", 1000))
        pesticides = float(data.get("pesticides", 10000))
        temperature = float(data.get("temperature", 24))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Invalid numerical parameters"}), 400

    try:
        if not area or not area_encoder or area not in area_encoder.classes_:
            return jsonify({"success": False, "error": "Invalid area or country"}), 400

        if not crop:
            return jsonify({"success": False, "error": "Missing crop variety parameter"}), 400

        resolution = resolve_crop(crop, manual_category=manual_category)
        archetype = resolution["archetype"]
        factor = resolution["scaling_factor"]

        if not crop_encoder or archetype not in crop_encoder.classes_:
            archetype = "Maize"

        enc_a = area_encoder.transform([area])[0]
        enc_c = crop_encoder.transform([archetype])[0]

        inp = pd.DataFrame([[enc_a, enc_c, year, rainfall, pesticides, temperature]], columns=[
            "Area", "Item", "Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"
        ])
        raw_yield_hg = max(0.0, float(model.predict(inp)[0]))
        yield_hg_ha = round(raw_yield_hg * factor, 2)
        yield_kg_ha = round(yield_hg_ha * 0.1, 2)
        yield_tonnes_ha = round(yield_kg_ha / 1000.0, 3)

        if resolution["is_manual"]:
            pred_type = "Manual Crop Estimate"
        elif resolution["is_extended"]:
            pred_type = "Extended Agronomic Estimate"
        elif year > 2013:
            pred_type = "Future-Year Model Estimate"
        else:
            pred_type = "Historical Benchmark"

        return jsonify({
            "success": True,
            "data": {
                "area": area,
                "crop": resolution["crop"],
                "category": resolution["category"],
                "archetype": archetype,
                "year": year,
                "yield_hg_ha": yield_hg_ha,
                "yield_kg_ha": yield_kg_ha,
                "yield_tonnes_ha": yield_tonnes_ha,
                "prediction_type": pred_type,
                "is_manual": resolution["is_manual"],
                "is_extended": resolution["is_extended"],
                "is_core": resolution["is_core"],
                "source_label": resolution["source_label"]
            }
        })
    except Exception as e:
        logger.exception("api_predict failed: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/assistant", methods=["POST"])
def api_assistant():
    data = request.get_json() or {}
    country = data.get("country", "India")
    crop = data.get("crop", "Maize")
    year = int(data.get("year", 2026))
    rainfall = float(data.get("rainfall", 1000))
    temperature = float(data.get("temperature", 24))
    pesticides = float(data.get("pesticides", 15000))

    predicted_hg_ha = 0.0
    ensure_models()
    if area_encoder and country in area_encoder.classes_ and crop_encoder and crop in crop_encoder.classes_ and model:
        enc_a = area_encoder.transform([country])[0]
        enc_c = crop_encoder.transform([crop])[0]
        inp = pd.DataFrame([[enc_a, enc_c, year, rainfall, pesticides, temperature]], columns=[
            "Area", "Item", "Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"
        ])
        predicted_hg_ha = max(0.0, float(model.predict(inp)[0]))

    stats = metadata.get("crop_statistics", {}).get(crop, {})
    hist_avg = stats.get("mean_yield_hg_ha", None)

    advice = generate_farming_advice(country, crop, rainfall, temperature, pesticides, year, predicted_hg_ha, hist_avg)
    return jsonify({"success": True, "data": advice})

# ===============================
# Run Application
# ===============================

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)
