"""
Automated Test Suite for AgriSense
Tests all critical components: ML prediction, authentication, APIs, Assistant, Comparison, and History.
"""

import os
import sys
import unittest
import json

# Add root directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import app, init_db, get_db_connection
from utils.crops_data import search_crops, get_all_crops, get_crop_details
from utils.assistant import generate_farming_advice, get_response

class AgriSenseTestCase(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        init_db()

    def test_01_crop_search_exact_and_partial(self):
        """Test crop search helper for exact, prefix, and case-insensitive matches."""
        # Partial match
        res = search_crops("mai")
        self.assertTrue(res["match_found"])
        self.assertIn("Maize", res["results"])

        # Case-insensitive
        res = search_crops("RICE")
        self.assertTrue(res["match_found"])
        self.assertTrue(any("Rice" in c for c in res["results"]))

        # Non-existent crop
        res = search_crops("DragonFruitNotFound")
        self.assertFalse(res["match_found"])
        self.assertEqual(len(res["results"]), 0)

    def test_02_crop_details_from_dataset(self):
        """Verify that crop details load real FAO historical stats."""
        details = get_crop_details("Wheat")
        self.assertIsNotNone(details)
        self.assertEqual(details["crop"], "Wheat")
        self.assertIn("mean_yield_kg_ha", details["statistics"])
        self.assertGreater(details["statistics"]["records"], 1000)
        self.assertGreater(len(details["yearly_trend"]), 10)

    def test_03_prediction_math_conversions_and_future_year(self):
        """Verify ML prediction returns correct math: 1 hg/ha = 0.1 kg/ha and kg/ha / 1000 = tonnes/ha."""
        # Historical year (2005)
        payload_historical = {
            "area": "India",
            "crop": "Maize",
            "year": 2005,
            "rainfall": 1000.0,
            "temperature": 24.0,
            "pesticides": 15000.0
        }
        res = self.client.post("/api/predict", json=payload_historical)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()["data"]
        self.assertEqual(data["prediction_type"], "Historical Benchmark")
        self.assertAlmostEqual(data["yield_kg_ha"], round(data["yield_hg_ha"] * 0.1, 2), places=1)
        self.assertAlmostEqual(data["yield_tonnes_ha"], round(data["yield_kg_ha"] / 1000.0, 3), places=2)

        # Future year (2026)
        payload_future = {
            "area": "India",
            "crop": "Wheat",
            "year": 2026,
            "rainfall": 600.0,
            "temperature": 18.0,
            "pesticides": 12000.0
        }
        res_future = self.client.post("/api/predict", json=payload_future)
        self.assertEqual(res_future.status_code, 200)
        data_future = res_future.get_json()["data"]
        self.assertEqual(data_future["prediction_type"], "Future-Year Model Estimate")
        self.assertAlmostEqual(data_future["yield_kg_ha"], round(data_future["yield_hg_ha"] * 0.1, 2), places=1)

    def test_04_input_validation(self):
        """Ensure invalid inputs like negative rainfall, invalid countries, or empty crops are rejected."""
        # Negative rainfall
        response = self.client.post("/prediction", data={
            "area": "India",
            "crop": "Maize",
            "year": "2026",
            "rainfall": "-500",
            "temperature": "24",
            "pesticides": "10000"
        }, follow_redirects=True)
        self.assertIn(b"Average annual rainfall must be between", response.data)

        # Empty crop name
        response_empty = self.client.post("/prediction", data={
            "area": "India",
            "crop": "",
            "year": "2026",
            "rainfall": "1000",
            "temperature": "24",
            "pesticides": "10000"
        }, follow_redirects=True)
        self.assertIn(b"Please select a crop variety or enter a crop name manually", response_empty.data)

        # Invalid country
        response_country = self.client.post("/prediction", data={
            "area": "NonExistentCountry999",
            "crop": "Maize",
            "year": "2026",
            "rainfall": "1000",
            "temperature": "24",
            "pesticides": "10000"
        }, follow_redirects=True)
        self.assertIn(b"not in the training dataset", response_country.data)

    def test_04b_extended_crop_prediction(self):
        """Verify that extended catalog crops (e.g. Cotton, Sugarcane, Barley) calculate calibrated yield properly."""
        response = self.client.post("/prediction", data={
            "area": "India",
            "crop": "Cotton",
            "year": "2026",
            "rainfall": "900",
            "temperature": "28",
            "pesticides": "15000"
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Cotton", response.data)
        self.assertIn(b"Extended Agronomic Estimate", response.data)
        self.assertIn(b"kg/ha", response.data)

    def test_04c_manual_crop_entry(self):
        """Verify that manually entered custom crops (e.g. Dragon Fruit) calculate calibrated yield without error."""
        response = self.client.post("/prediction", data={
            "area": "India",
            "crop": "Dragon Fruit",
            "manual_category": "Fruits & Plantation",
            "is_manual_mode": "1",
            "year": "2026",
            "rainfall": "1200",
            "temperature": "27",
            "pesticides": "8000"
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Dragon Fruit", response.data)
        self.assertIn(b"Manual Custom Crop Estimate", response.data)
        self.assertIn(b"Agronomic Calibration Notice", response.data)

    def test_04d_api_predict_extended_and_manual(self):
        """Verify /api/predict handles both extended crops and manual crops via API."""
        # Extended crop (Tomatoes)
        res_ext = self.client.post("/api/predict", json={
            "area": "India",
            "crop": "Tomatoes",
            "year": 2026,
            "rainfall": 800.0,
            "temperature": 22.0,
            "pesticides": 10000.0
        })
        self.assertEqual(res_ext.status_code, 200)
        data_ext = res_ext.get_json()["data"]
        self.assertEqual(data_ext["crop"], "Tomatoes")
        self.assertEqual(data_ext["prediction_type"], "Extended Agronomic Estimate")
        self.assertGreater(data_ext["yield_kg_ha"], 0)

        # Manual crop (Quinoa)
        res_man = self.client.post("/api/predict", json={
            "area": "India",
            "crop": "Quinoa",
            "manual_category": "Cereals & Grains",
            "year": 2026,
            "rainfall": 600.0,
            "temperature": 18.0,
            "pesticides": 5000.0
        })
        self.assertEqual(res_man.status_code, 200)
        data_man = res_man.get_json()["data"]
        self.assertEqual(data_man["crop"], "Quinoa")
        self.assertEqual(data_man["prediction_type"], "Manual Crop Estimate")
        self.assertTrue(data_man["is_manual"])

    def test_05_assistant_with_country_and_ml_prediction(self):
        """Verify the Smart Farming Assistant handles Country + ML predicted yield + advice generation."""
        advice = generate_farming_advice(
            country="India",
            crop="Cassava",
            rainfall=1100,
            temperature=22,
            pesticides=20000,
            year=2026,
            predicted_yield_hg_ha=150000.0,
            historical_avg_hg_ha=140000.0
        )
        self.assertTrue(advice["success"])
        self.assertEqual(advice["crop"], "Cassava")
        self.assertEqual(advice["country"], "India")
        self.assertIn("Crop Suitability", advice["advice_text"])
        self.assertIn("Rainfall", advice["advice_text"])
        self.assertIn("Integrated Pest Management", advice["advice_text"])
        self.assertEqual(advice["predicted_kg_ha"], 15000.0)

    def test_06_dashboard_chart_data_api(self):
        """Verify that dashboard chart data API returns all 8 required charts with data."""
        res = self.client.get("/api/chart-data")
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertTrue(json_data["success"])
        charts = json_data["data"]
        
        required_charts = [
            "chart1_crop_yield",
            "chart2_yearly_trend",
            "chart3_feature_importance",
            "chart4_top_countries",
            "chart5_rainfall_vs_yield",
            "chart6_temp_vs_yield",
            "chart7_pest_vs_yield",
            "chart8_yield_spread"
        ]
        for c in required_charts:
            self.assertIn(c, charts, f"Missing required chart {c}")
            self.assertGreater(len(charts[c]["labels"]), 0, f"Chart {c} has empty labels")

    def test_07_auth_and_prediction_history(self):
        """Test registration, secure login, prediction saving, and history retrieval."""
        import uuid
        test_email = f"testuser_{uuid.uuid4().hex[:6]}@example.com"
        test_user = f"farmer_{uuid.uuid4().hex[:4]}"
        test_pass = "SecurePass123"

        # Register
        reg_res = self.client.post("/signup", data={
            "username": test_user,
            "email": test_email,
            "password": test_pass
        }, follow_redirects=True)
        self.assertIn(b"Registration successful", reg_res.data)

        # Login
        login_res = self.client.post("/login", data={
            "email": test_email,
            "password": test_pass
        }, follow_redirects=True)
        self.assertIn(b"Welcome back", login_res.data)

        # Make prediction while logged in
        pred_res = self.client.post("/prediction", data={
            "area": "India",
            "crop": "Rice, paddy",
            "year": "2026",
            "rainfall": "1500",
            "temperature": "27",
            "pesticides": "15000"
        }, follow_redirects=True)
        self.assertEqual(pred_res.status_code, 200)
        self.assertIn(b"Rice, paddy", pred_res.data)

        # Verify history displays the prediction
        hist_res = self.client.get("/history")
        self.assertIn(b"Rice, paddy", hist_res.data)
        self.assertIn(b"India", hist_res.data)

    def test_08_multi_crop_comparison(self):
        """Test multi-crop comparison route with multiple crops under identical conditions."""
        res = self.client.post("/compare", data={
            "area": "India",
            "year": "2026",
            "rainfall": "1000",
            "temperature": "24",
            "pesticides": "15000",
            "selected_crops": ["Maize", "Wheat", "Rice, paddy"]
        })
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Maize", res.data)
        self.assertIn(b"Wheat", res.data)
        self.assertIn(b"Rice, paddy", res.data)
        self.assertIn(b"Yield Comparison", res.data)

    def test_09_crop_explorer_and_detail_routes(self):
        """Test browsing all crops and accessing single crop analytical deep dive."""
        res = self.client.get("/crops")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Crop Explorer", res.data)
        self.assertIn(b"Potatoes", res.data)

        # Single crop detail
        detail_res = self.client.get("/crops/Potatoes")
        self.assertEqual(detail_res.status_code, 200)
        self.assertIn(b"Potatoes", detail_res.data)
        self.assertIn(b"Historical Yield Trend", detail_res.data)

    def test_10_assistant_chat_qa_and_about_page(self):
        """Test conversational Q&A and about page rendering."""
        res = self.client.post("/assistant", data={
            "action_type": "chat_qa",
            "question": "What is IPM in agriculture?"
        })
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Integrated Pest Management", res.data)

        about_res = self.client.get("/about")
        self.assertEqual(about_res.status_code, 200)
        self.assertIn(b"RandomForestRegressor", about_res.data)
        self.assertIn(b"Food and Agriculture Organization", about_res.data)

    def test_14_ensure_models_auto_recovery(self):
        """Verify that ensure_models restores models and encoders if uninitialized."""
        import app as app_module
        # Simulate uninitialized state
        orig_model = app_module.model
        orig_area_encoder = app_module.area_encoder
        orig_crop_encoder = app_module.crop_encoder

        app_module.model = None
        app_module.area_encoder = None
        app_module.crop_encoder = None

        self.assertTrue(app_module.ensure_models())
        self.assertIsNotNone(app_module.model)
        self.assertIsNotNone(app_module.area_encoder)
        self.assertIsNotNone(app_module.crop_encoder)

        # Confirm prediction works immediately after recovery
        res = self.client.post("/api/predict", json={
            "area": "India",
            "crop": "Maize",
            "year": 2026,
            "rainfall": 1000.0,
            "temperature": 24.0,
            "pesticides": 15000.0
        })
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["success"])


if __name__ == "__main__":
    unittest.main()
