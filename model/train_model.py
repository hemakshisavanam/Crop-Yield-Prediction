"""
AgriSense - Machine Learning Training Pipeline
Dataset: dataset/yield_df.csv
Target: hg/ha_yield
Features: Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "yield_df.csv")
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    total_raw_records = len(df)
    print(f"Total raw records loaded: {total_raw_records}")

    # Validate required columns
    required_cols = [
        "Area", "Item", "Year", "hg/ha_yield",
        "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Clean data: drop unwanted index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Drop missing values
    df = df.dropna(subset=required_cols)

    # Deduplicate rows
    df = df.drop_duplicates()
    clean_records_count = len(df)
    print(f"Cleaned records after removing duplicates: {clean_records_count}")

    # Fit Label Encoders
    area_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()

    unique_crops = sorted([str(c) for c in df["Item"].unique()])
    unique_areas = sorted([str(a) for a in df["Area"].unique()])

    # Per-Crop historical statistics from clean data
    crop_stats = {}
    for crop in unique_crops:
        crop_sub = df[df["Item"] == crop]
        yields = crop_sub["hg/ha_yield"]
        crop_stats[crop] = {
            "records": int(len(crop_sub)),
            "countries_count": int(crop_sub["Area"].nunique()),
            "mean_yield_hg_ha": round(float(yields.mean()), 2),
            "mean_yield_kg_ha": round(float(yields.mean() * 0.1), 2),
            "mean_yield_tonnes_ha": round(float(yields.mean() * 0.0001), 3),
            "min_yield_hg_ha": round(float(yields.min()), 2),
            "max_yield_hg_ha": round(float(yields.max()), 2),
            "std_yield_hg_ha": round(float(yields.std()), 2) if len(yields) > 1 else 0.0,
            "avg_rainfall_mm": round(float(crop_sub["average_rain_fall_mm_per_year"].mean()), 1),
            "avg_temp_c": round(float(crop_sub["avg_temp"].mean()), 1)
        }

    df["Area"] = area_encoder.fit_transform(df["Area"].astype(str))
    df["Item"] = crop_encoder.fit_transform(df["Item"].astype(str))

    feature_cols = [
        "Area",
        "Item",
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp"
    ]

    X = df[feature_cols]
    y = df["hg/ha_yield"].astype(float)

    # 80/20 train/test split with fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")

    # Train Random Forest Regressor
    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Model Evaluation on Test Set
    predictions = model.predict(X_test)
    r2 = float(r2_score(y_test, predictions))
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    print(f"\n--- Model Evaluation Results ---")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE:      {mae:.2f} hg/ha ({mae * 0.1:.2f} kg/ha)")
    print(f"RMSE:     {rmse:.2f} hg/ha ({rmse * 0.1:.2f} kg/ha)")

    # Feature Importance Mapping
    feature_names_readable = [
        "Country / Area",
        "Crop Type",
        "Year",
        "Average Rainfall (mm/year)",
        "Pesticides (tonnes)",
        "Average Temperature (°C)"
    ]
    raw_feature_names = [
        "Area",
        "Item",
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp"
    ]
    feature_importances = {
        name: round(float(imp), 4)
        for name, imp in zip(raw_feature_names, model.feature_importances_)
    }
    print(f"Feature Importances: {feature_importances}")

    # Range metadata for input validation and context
    ranges = {
        "year": {
            "min": int(df["Year"].min()),
            "max": int(df["Year"].max())
        },
        "rainfall_mm": {
            "min": float(df["average_rain_fall_mm_per_year"].min()),
            "max": float(df["average_rain_fall_mm_per_year"].max()),
            "mean": round(float(df["average_rain_fall_mm_per_year"].mean()), 1)
        },
        "pesticides_tonnes": {
            "min": float(df["pesticides_tonnes"].min()),
            "max": float(df["pesticides_tonnes"].max()),
            "mean": round(float(df["pesticides_tonnes"].mean()), 1)
        },
        "avg_temp_c": {
            "min": float(df["avg_temp"].min()),
            "max": float(df["avg_temp"].max()),
            "mean": round(float(df["avg_temp"].mean()), 1)
        },
        "yield_hg_ha": {
            "min": float(df["hg/ha_yield"].min()),
            "max": float(df["hg/ha_yield"].max()),
            "mean": round(float(df["hg/ha_yield"].mean()), 1)
        }
    }

    metadata = {
        "model_name": "RandomForestRegressor",
        "n_estimators": 100,
        "dataset_name": "yield_df.csv",
        "dataset_source": "Food and Agriculture Organization (FAO) Agricultural Statistics",
        "raw_records": total_raw_records,
        "clean_records": clean_records_count,
        "training_records": len(X_train),
        "test_records": len(X_test),
        "metrics": {
            "r2_score": round(r2, 4),
            "mae_hg_ha": round(mae, 2),
            "mae_kg_ha": round(mae * 0.1, 2),
            "rmse_hg_ha": round(rmse, 2),
            "rmse_kg_ha": round(rmse * 0.1, 2)
        },
        "features": raw_feature_names,
        "feature_names_readable": feature_names_readable,
        "feature_importances": feature_importances,
        "crops_count": len(unique_crops),
        "areas_count": len(unique_areas),
        "crops": unique_crops,
        "areas": unique_areas,
        "ranges": ranges,
        "crop_statistics": crop_stats
    }

    # Save artifacts
    crop_yield_model_path = os.path.join(model_dir, "crop_yield_model.pkl")
    crop_model_path = os.path.join(model_dir, "crop_model.pkl")
    area_encoder_path = os.path.join(model_dir, "area_encoder.pkl")
    crop_encoder_path = os.path.join(model_dir, "crop_encoder.pkl")
    metadata_path = os.path.join(model_dir, "metadata.json")

    joblib.dump(model, crop_yield_model_path)
    joblib.dump(model, crop_model_path)
    joblib.dump(area_encoder, area_encoder_path)
    joblib.dump(crop_encoder, crop_encoder_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nTraining complete! Artifacts saved:")
    print(f"  [OK] {crop_yield_model_path}")
    print(f"  [OK] {crop_model_path}")
    print(f"  [OK] {area_encoder_path}")
    print(f"  [OK] {crop_encoder_path}")
    print(f"  [OK] {metadata_path}")

if __name__ == "__main__":
    train()