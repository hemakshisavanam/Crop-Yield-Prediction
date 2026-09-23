# AgriSense — AI-Based Crop Yield Prediction and Smart Farming Assistant

> **AI-Based Crop Yield Prediction and Smart Farming Assistant**  
> *A Machine-Learning & Generative-AI Agricultural Decision-Support System*

---

## Overview

**AgriSense** is an enterprise-grade agricultural decision-support web platform engineered to bridge the gap between historical agricultural records and predictive analytics. By harnessing historical agricultural data from the Food and Agriculture Organization (FAO), advanced machine learning (Random Forest Regressor), and Google Gemini Generative AI, AgriSense enables farmers, agronomists, and agricultural researchers to estimate crop yields, simulate future climate scenarios, benchmark crop suitability, and access tailored agronomic guidance.

This platform was developed to serve as a comprehensive **B.Tech Final-Year Engineering Project**, portfolio showcase, hackathon demonstration, and practical decision-support tool.

---

## Problem Statement

Global agricultural productivity faces severe challenges from erratic rainfall patterns, temperature anomalies, soil degradation, and unbalanced pesticide applications. Farmers often lack accessible predictive tools to forecast crop yield under changing environmental conditions, making informed decisions on crop selection, water management, and pest control difficult. Existing systems are often either overly simplified calculators or opaque black-box models that fail to explain predictions or provide practical, farmer-friendly guidance.

---

## Proposed Solution

AgriSense addresses these challenges through a unified, full-stack platform:
1. **Accurate Yield Forecasting**: An empirical Machine Learning regression pipeline trained on 25,932 clean historical FAO observation records.
2. **Future Scenario Simulation**: Statistically grounded future-year predictions (2024–2035) with analytical context explaining extrapolation beyond the historical baseline (1990–2013).
3. **Multi-Crop Benchmarking**: Comparative analysis of alternative crops under identical environmental conditions to determine optimal crop suitability.
4. **Smart Farming Assistant**: Country-specific, crop-tailored agronomic guidance combining numerical ML yield predictions with Google Gemini natural language advice and a robust offline agronomic fallback engine.
5. **Transparency & Scientific Honesty**: Full disclosure of model accuracy metrics (R² = 0.9833), dataset boundaries, and agronomic limitations without fabricated claims.

---

## Objectives

* **High-Accuracy Modeling**: Train and evaluate a high-accuracy regression model (R² > 98%) to model complex interactions between geography, crop variety, rainfall, temperature, and pesticide volume.
* **Accessible Usability**: Provide an intuitive web interface with dynamic crop search, interactive sliders, live unit conversions (`hg/ha`, `kg/ha`, `tonnes/ha`), and responsive mobile design.
* **Farmer-Friendly Advisory**: Provide structured, actionable advice covering crop suitability, irrigation scheduling, heat stress mitigation, and Integrated Pest Management (IPM).
* **Data Transparency**: Maintain a dynamic analytics dashboard with 8 real-data interactive visualizations and a user-isolated prediction history log.

---

## Features

| Feature | Description |
| :--- | :--- |
| **Searchable Dynamic Crop Selector** | Instant partial, prefix, and case-insensitive matching across all 10 verified FAO crops with quick-selection pills. |
| **Multivariable Yield Prediction** | Calculates harvest outputs in `hg/ha`, `kg/ha` (`hg/ha × 0.1`), and `tonnes/ha` (`kg/ha ÷ 1000`). |
| **Future-Year Estimation** | Simulates scenarios for future seasons (e.g. 2024–2035) with clear analytical indicators and notices. |
| **Explore Another Crop** | Fast one-click workflow that preserves country, year, rainfall, temperature, and pesticide parameters while allowing the user to select another crop. |
| **Multi-Crop Comparison** | Side-by-side comparative table and Chart.js bar chart evaluating multiple crops under identical conditions. |
| **Crop Explorer (`/crops`)** | Deep dive into historical yields, standard deviations, top producing regions, and yearly trends for each crop. |
| **Smart Farming Assistant** | Input Country + Crop + Weather &rarr; ML predicted yield &rarr; Google Gemini generative guidance (with rule-based fallback). |
| **Analytics Dashboard** | 8 dynamic interactive Chart.js visualizations covering feature importances, climate-yield distributions, and global rankings. |
| **Prediction History** | Persistent SQLite activity log with search, crop filtering, and record deletion. |
| **Secure Authentication** | User registration and login protected with modern `werkzeug.security` password hashing (`scrypt`/`PBKDF2`). |

---

## System Architecture

```text
                                  USER INTERFACE
    (HTML5 • CSS3 Modern Design System • Bootstrap 5.3 • Chart.js • Vanilla JS)
                                        │
                                        ▼
                                 FLASK BACKEND
                (Routing • Session Auth • REST APIs • Input Validation)
                                        │
                   ┌────────────────────┴────────────────────┐
                   ▼                                         ▼
            SQLITE DATABASE                         ML & ANALYTICS PIPELINE
         (users.db: Users, History)              (Joblib • Scikit-Learn • Pandas)
                                                             │
                                        ┌────────────────────┴────────────────────┐
                                        ▼                                         ▼
                               RANDOM FOREST MODEL                        GOOGLE GEMINI AI
                            (crop_yield_model.pkl)                       (google-genai SDK)
                                        │                                         │
                                        ▼                                         ▼
                            Numerical Yield Output                   Personalized Agronomy Advice
                           (hg/ha • kg/ha • tonnes/ha)              (Suitability, Water, IPM, Soil)
```

---

## Dataset

* **Source**: Food and Agriculture Organization (FAO) Agricultural Statistics Database (`yield_df.csv`).
* **Raw Records**: 28,242 rows.
* **Cleaned Records**: 25,932 rows (after removing duplicate records and index artifacts).
* **Temporal Span**: 1990 &ndash; 2013 (24 years of documented agricultural observations).
* **Geographic Coverage**: 101 countries and agricultural territories (Albania, India, United States, Brazil, etc.).
* **Target Variable**: `hg/ha_yield` (hectograms per hectare).
* **Observed Features**:
  1. `Area`: Sovereign nation or territory (categorical).
  2. `Item`: Crop variety (categorical).
  3. `Year`: Observation year (integer).
  4. `average_rain_fall_mm_per_year`: Annual precipitation in millimeters (51 &ndash; 3,240 mm).
  5. `pesticides_tonnes`: Annual national pesticide volume in metric tonnes (0.04 &ndash; 367,778 tonnes).
  6. `avg_temp`: Average annual temperature in degrees Celsius (1.3 &ndash; 30.65 °C).

### Verified Crops Catalog (10 Crops)
1. **Cassava**
2. **Maize**
3. **Plantains and others**
4. **Potatoes**
5. **Rice, paddy**
6. **Sorghum**
7. **Soybeans**
8. **Sweet potatoes**
9. **Wheat**
10. **Yams**

---

## Machine Learning

The core estimation engine utilizes a **Random Forest Regressor** trained using an 80/20 train/test split with a fixed random state (`random_state=42`).

### Verified Evaluation Metrics (Actual Generated Results)

| Metric | Target Unit (`hg/ha`) | Converted Unit (`kg/ha`) |
| :--- | :--- | :--- |
| **Coefficient of Determination (R²)** | **0.9833 (98.33%)** | **0.9833** |
| **Mean Absolute Error (MAE)** | **4,221.72 hg/ha** | **422.17 kg/ha** |
| **Root Mean Squared Error (RMSE)** | **10,999.90 hg/ha** | **1,099.99 kg/ha** |
| **Training Records** | 20,745 observations | — |
| **Testing Records** | 5,187 observations | — |

### Feature Importance Breakdown

* **Crop Variety (`Item`)**: **59.81%** (Dominant factor; baseline yield differs significantly across species, e.g. root tubers vs cereal grains).
* **Pesticide Volume (`pesticides_tonnes`)**: **11.83%**
* **Average Temperature (`avg_temp`)**: **10.28%**
* **Annual Rainfall (`average_rain_fall_mm_per_year`)**: **8.08%**
* **Country / Region (`Area`)**: **6.90%**
* **Production Year (`Year`)**: **3.10%**

---

## Future-Year Prediction

AgriSense allows users to input historical years, recent years, and future years (e.g. 2024, 2025, 2026, 2030, 2035).

### How Future Predictions Work
* The model takes the entered year directly through the trained pipeline.
* Extrapolations utilize historical technological and yield trends learned across 1990–2013 combined with the entered rainfall, temperature, and pesticide conditions.
* **Never Fabricated**: Future years are explicitly tagged in the UI with a distinct amber badge:
  ```text
  Prediction Type: Future-Year Model Estimate
  Training Baseline: 1990–2013
  ```
* **Notice Displayed**:
  > *Future-year predictions are estimates based on patterns learned from historical agricultural data and the conditions entered by the user. They are not guaranteed actual future yields.*

---

## Crop Search

* Search input supports exact match, partial match, and case-insensitive search (e.g., `mai` &rarr; `Maize`, `rice` &rarr; `Rice, paddy`).
* Available crops are retrieved dynamically from `metadata.json` and the trained `crop_encoder`.
* If an unsupported crop is requested, the system politely alerts the user that the crop is not directly represented in the training dataset and invites them to select from the verified 10 crops.

---

## Crop Comparison

The **Compare Crops** module (`/compare`) allows users to calibrate a single geographic area and climate profile (e.g. India, 2026, 1000 mm rainfall, 24°C, 15,000 tonnes pesticides) and select multiple crops via checkboxes.

The system generates:
* An interactive side-by-side comparative bar chart (predicted yield vs. historical benchmark).
* A structured table displaying raw `hg/ha`, converted `kg/ha`, `tonnes/ha`, historical regional averages, and variance.

---

## Smart Farming Assistant

The **Smart Farming Assistant** (`/assistant`) combines numerical machine-learning output with Generative AI:
1. **User Input**: Country + Crop + Year + Rainfall + Temperature + Pesticides.
2. **Machine Learning Step**: The Flask backend runs the input through the Random Forest model to calculate the numerical yield.
3. **AI Generation Step**: The backend passes the country, crop, weather metrics, and predicted yield to the **Google Gemini API** (`google-genai` SDK using `gemini-2.5-flash`).
4. **Structured Guidance**:
   * 🌱 **Crop Suitability & Climate Fit**: Evaluates thermal and moisture suitability.
   * 🌧 **Rainfall & Water Management**: Irrigation scheduling and field drainage strategies.
   * 🌡 **Temperature Impact**: Heat stress mitigation and frost protection.
   * 🐛 **Integrated Pest Management (IPM)**: Low-chemical, biological pest control for specific crop pests.
   * 🌾 **Practical Farming Recommendations**: Sowing windows, land preparation, and balanced NPK fertilizer management.
   * 📈 **Predicted Yield Interpretation**: Plain-language explanation comparing the yield with regional baselines.
5. **Data-Driven Offline Fallback**: If no Gemini API key is configured or network is unavailable, a comprehensive rule-based agronomic engine automatically provides the exact same structured sections based on published agronomic thresholds.

---

## Dashboard

The Analytics Dashboard (`/dashboard`) dynamically renders 8 Chart.js visualizations:
1. **Average Yield by Crop** (kg/ha mean comparison).
2. **Historical Global Yield Trend** (1990–2013 yearly progress).
3. **Random Forest Feature Importance** (percentage weights).
4. **Top 10 Agricultural Regions by Yield** (highest producing countries).
5. **Annual Rainfall Range vs. Harvest Yield** (precipitation bins).
6. **Temperature Range vs. Harvest Yield** (thermal bins).
7. **Pesticide Volume vs. Harvest Yield** (chemical usage bins).
8. **Crop Yield Spread** (minimum, mean, and maximum yield boundaries).

---

## Authentication

* User accounts are managed in SQLite (`users.db`).
* Passwords are securely hashed using `werkzeug.security` (`generate_password_hash` and `check_password_hash` using `scrypt`/`PBKDF2`).
* User session isolation ensures users only see and manage their own prediction history.
* Secrets and API keys are managed through `.env` and excluded from git version control.

---

## Technology Stack

### Backend
* **Python 3.11+**
* **Flask 3.1.0** (Web framework)
* **Scikit-Learn 1.6.1** (Random Forest Regressor & Label Encoding)
* **Pandas 2.2.3 & NumPy 2.2.2** (Data processing)
* **Joblib 1.4.2** (Model serialization)
* **google-genai 1.0+** (Official Google Gemini Python SDK)
* **python-dotenv 1.0.1** (Environment variables)
* **SQLite3** (Lightweight relational database)

### Frontend
* **HTML5 & Semantic Structure**
* **Vanilla CSS3 Custom Design System** (Glassmorphism, animations, responsive layout)
* **Bootstrap 5.3.3** (Grid & utilities)
* **Chart.js 4.4.1** (Interactive canvas charts)
* **FontAwesome 6.5.1** (Vector icons)
* **Google Fonts** (Outfit for headings, Inter for body text)

---

## Project Structure

```text
crop_yield_21/
├── app.py                     # Main Flask application with routes and REST APIs
├── requirements.txt           # Python dependencies
├── .env.example               # Template for environment configuration
├── .env                       # Local environment variables (excluded in .gitignore)
├── .gitignore                 # Excludes .env, caches, and artifacts
├── users.db                   # SQLite database (Users & Prediction History)
│
├── dataset/
│   └── yield_df.csv           # Verified FAO agricultural dataset (28,242 rows)
│
├── model/
│   ├── train_model.py         # ML training script with deduplication & evaluation
│   ├── crop_yield_model.pkl   # Serialized Random Forest model
│   ├── crop_model.pkl         # Compatibility model alias
│   ├── area_encoder.pkl       # LabelEncoder for 101 countries
│   ├── crop_encoder.pkl       # LabelEncoder for 10 crops
│   └── metadata.json          # Verified training metrics, ranges & crop statistics
│
├── utils/
│   ├── assistant.py           # Gemini Generative AI & rule-based agronomic fallback
│   └── crops_data.py          # Data caching, search helper, and chart aggregations
│
├── templates/
│   ├── base.html              # Master layout with navbar and footer
│   ├── home.html              # Modern landing page with feature showcase
│   ├── predict.html           # Prediction form with searchable crop selector
│   ├── result.html            # Prediction result card with conversions & context
│   ├── crops.html             # Crop Explorer directory
│   ├── crop_detail.html       # Single crop analytical deep dive with trend chart
│   ├── compare.html           # Multi-crop comparison table and bar chart
│   ├── assistant.html         # Smart Farming Assistant with Country dropdown
│   ├── dashboard.html         # Analytics dashboard with 8 Chart.js charts
│   ├── history.html           # Prediction history table with search & delete
│   ├── login.html             # User login card
│   ├── signup.html            # User registration card
│   └── about.html             # Academic project overview & architecture
│
├── static/
│   ├── css/
│   │   └── style.css          # Modern agricultural SaaS CSS design system
│   └── js/
│       └── script.js          # Dynamic crop search, Chart.js, and converters
│
└── tests/
    └── test_agrisense.py      # Automated test suite (10 unit & integration tests)
```

---

## Installation

### Prerequisites
* Python 3.10, 3.11, or 3.12 installed.
* Git (optional).

### 1. Clone or Open Project Directory
```bash
cd c:\Users\savan\OneDrive\Desktop\crop_yield_21
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows Command Prompt:
.\venv\Scripts\activate.bat
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and optionally set your `GEMINI_API_KEY`:
```env
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=agrisense_secure_session_key_2026

# Optional: Add Google Gemini API Key for live AI responses
# Get free key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Note: If no key is set, AgriSense automatically uses its built-in rule-based agronomic intelligence engine.)*

---

## Running the Application

Start the Flask server:
```bash
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## Training the Model

To retrain the Random Forest model and regenerate `metadata.json`:
```bash
python model/train_model.py
```
Output:
```text
Loading dataset from: dataset/yield_df.csv
Total raw records loaded: 28242
Cleaned records after removing duplicates: 25932
Training rows: 20745, Testing rows: 5187
Training RandomForestRegressor model...

--- Model Evaluation Results ---
R² Score: 0.9833
MAE:      4221.72 hg/ha (422.17 kg/ha)
RMSE:     10999.90 hg/ha (1099.99 kg/ha)
Feature Importances: {'Area': 0.069, 'Item': 0.5981, 'Year': 0.031, 'average_rain_fall_mm_per_year': 0.0808, 'pesticides_tonnes': 0.1183, 'avg_temp': 0.1028}

Training complete! Artifacts saved:
  [OK] model/crop_yield_model.pkl
  [OK] model/crop_model.pkl
  [OK] model/area_encoder.pkl
  [OK] model/crop_encoder.pkl
  [OK] model/metadata.json
```

---

## Running Automated Tests

Run the comprehensive 10-test automated suite:
```bash
python tests/test_agrisense.py
```
Output:
```text
..........
----------------------------------------------------------------------
Ran 10 tests in 1.144s

OK
```

---

## API Documentation

AgriSense provides structured REST APIs returning consistent JSON envelopes:

### 1. Crop Search
* **Endpoint**: `GET /api/crops/search?q=<query>`
* **Sample Response**:
  ```json
  {
    "success": true,
    "data": {
      "match_found": true,
      "results": ["Maize"],
      "best_match": "Maize",
      "query": "mai"
    }
  }
  ```

### 2. Yield Prediction
* **Endpoint**: `POST /api/predict`
* **Payload**:
  ```json
  {
    "area": "India",
    "crop": "Maize",
    "year": 2026,
    "rainfall": 1000,
    "temperature": 24,
    "pesticides": 15000
  }
  ```
* **Sample Response**:
  ```json
  {
    "success": true,
    "data": {
      "area": "India",
      "crop": "Maize",
      "year": 2026,
      "yield_hg_ha": 25140.2,
      "yield_kg_ha": 2514.02,
      "yield_tonnes_ha": 2.514,
      "prediction_type": "Future-Year Model Estimate"
    }
  }
  ```

### 3. Dashboard Chart Data
* **Endpoint**: `GET /api/chart-data`
* **Description**: Returns pre-aggregated real data for all 8 dashboard charts.

### 4. Smart Assistant Advisory
* **Endpoint**: `POST /api/assistant`
* **Payload**: `{ "country": "India", "crop": "Cassava", "year": 2026, "rainfall": 1100, "temperature": 22, "pesticides": 20000 }`
* **Response**: Returns structured advice with crop suitability, irrigation, thermal stress, and IPM.

---

## Limitations

* **Historical Scope**: Training data covers 1990–2013; future-year predictions are mathematical extrapolations of learned patterns.
* **Aggregated Macro Metrics**: Pesticide figures reflect national-level FAO annual usage volumes rather than localized farm-scale dosage per acre.
* **Micro-Climate Blindness**: Does not currently ingest live Doppler radar, frost warnings, or immediate hail forecasts.
* **Soil Chemistry**: Precision variables like soil pH, nitrogen-phosphorus-potassium (NPK) ratios, and microbial activity are not in the FAO global dataset.

---

## Future Enhancements

* Integration with real-time weather APIs (OpenWeatherMap / WeatherAPI).
* Satellite-based Normalized Difference Vegetation Index (NDVI) monitoring.
* Soil testing integration (pH, organic carbon, NPK balance).
* IoT edge sensor support for automated soil moisture telemetry.
* Mobile application build using Flutter.
* Support for vernacular local languages (Hindi, Spanish, French, Swahili).

---

## Disclaimer

> **Scientific & Legal Notice:** AgriSense provides machine-learning-based yield estimates and general agricultural guidance based on historical Food and Agriculture Organization (FAO) data and user-provided inputs. Predictions are mathematical estimations and should not be treated as guaranteed real-world outcomes, official crop insurance ratings, or certified professional agronomic advice.