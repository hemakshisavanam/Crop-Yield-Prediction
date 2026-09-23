"""
AgriSense - Smart Farming Assistant
Combines ML Numerical Prediction with Google Gemini Generative AI and Data-Driven Agronomic Rules.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

# Rule-based agronomic benchmarks derived from agronomic literature and dataset averages
CROP_AGRONOMIC_PROFILES = {
    "Cassava": {
        "opt_rain": (1000, 1500),
        "opt_temp": (25, 29),
        "soil": "Well-drained light sandy loam to clay loam; tolerant of low fertility.",
        "pests": "Cassava mosaic disease (whiteflies), green spider mites, mealybugs.",
        "water": "Highly drought-tolerant once established; avoid waterlogged soils.",
        "ipm": "Use disease-free stem cuttings, biological control (neoseiulus predatory mites), crop rotation."
    },
    "Maize": {
        "opt_rain": (600, 1200),
        "opt_temp": (18, 27),
        "soil": "Deep, fertile, well-aerated rich loamy soil with neutral pH (5.8-7.0).",
        "pests": "Fall armyworm, stem borers, earworms, leaf blight.",
        "water": "Critical moisture requirement during tasseling and silking stages.",
        "ipm": "Pheromone traps, neem-based sprays, biocontrol (Trichogramma), intercropping with legumes."
    },
    "Plantains and others": {
        "opt_rain": (1200, 2200),
        "opt_temp": (26, 30),
        "soil": "Humus-rich, deep, well-drained alluvial soil; highly sensitive to stagnant water.",
        "pests": "Banana weevil, black sigatoka leaf fungus, nematodes.",
        "water": "Consistent moisture year-round with efficient surface drainage.",
        "ipm": "Resistant cultivars, de-trashing diseased leaves, solarization of planting suckers."
    },
    "Potatoes": {
        "opt_rain": (500, 800),
        "opt_temp": (15, 20),
        "soil": "Loose, friable, acidic to neutral sandy loam (pH 5.2-6.5) to facilitate tuber growth.",
        "pests": "Late blight (Phytophthora infestans), Colorado potato beetle, aphids.",
        "water": "Even moisture through tuber bulking; avoid excessive late-season wetting.",
        "ipm": "Certified seed tubers, ridge earthing-up, copper fungicides on forecast, avoid night overhead irrigation."
    },
    "Rice, paddy": {
        "opt_rain": (1200, 2000),
        "opt_temp": (22, 32),
        "soil": "Heavy clay loam or silty clay with high water-holding capacity (submerged/flooded beds).",
        "pests": "Brown planthopper, stem borer, rice blast fungus, bacterial blight.",
        "water": "Standing water of 2-5 cm during vegetative and reproductive stages; drain before harvest.",
        "ipm": "Alternate Wetting and Drying (AWD), duck/fish co-culture, light traps, balanced potassium to resist blast."
    },
    "Sorghum": {
        "opt_rain": (450, 750),
        "opt_temp": (25, 32),
        "soil": "Adaptable to diverse soils, highly tolerant of alkaline and poor saline soils.",
        "pests": "Shoot fly, stem borer, sorghum midge, head bugs.",
        "water": "Extremely drought-resilient; efficient water-use C4 cereal grain.",
        "ipm": "Early synchronized sowing, seed treatment with imidacloprid/neem, botanical extracts."
    },
    "Soybeans": {
        "opt_rain": (600, 900),
        "opt_temp": (20, 28),
        "soil": "Warm, well-aerated fertile loams; inoculate with Bradyrhizobium for nitrogen fixation.",
        "pests": "Pod borers, whiteflies, soybean rust, root-knot nematodes.",
        "water": "Adequate moisture vital during flowering and pod development phases.",
        "ipm": "Crop rotation with corn/grasses, beneficial insect refuges, targeted biological insecticides."
    },
    "Sweet potatoes": {
        "opt_rain": (750, 1000),
        "opt_temp": (21, 28),
        "soil": "Moderately sandy or well-drained loamy soil; heavy soils cause misshapen tubers.",
        "pests": "Sweetpotato weevil, whiteflies, stem borer.",
        "water": "Moderate moisture; excess water encourages excessive vine growth over root bulking.",
        "ipm": "Weevil pheromone traps, vine dipping before planting, prompt harvesting after tuber maturity."
    },
    "Wheat": {
        "opt_rain": (450, 750),
        "opt_temp": (14, 22),
        "soil": "Well-drained, medium-textured fertile loamy soil with neutral pH (6.0-7.5).",
        "pests": "Rusts (stripe, stem, leaf), aphids, Hessian fly, powdery mildew.",
        "water": "Crown root initiation (CRI), tillering, and grain-filling are critical watering stages.",
        "ipm": "Rust-resistant cultivars, seed treatment, scouting for aphids, timely sowing to avoid terminal heat stress."
    },
    "Yams": {
        "opt_rain": (1000, 1500),
        "opt_temp": (25, 30),
        "soil": "Deep, loose, organic-rich fertile soils on mounds or ridges to enable tuber expansion.",
        "pests": "Yam beetle, scale insects, anthracnose leaf spot, nematodes.",
        "water": "Steady distribution during first 4 months of canopy and tuber initiation.",
        "ipm": "Hot-water sett treatment, staking for canopy aeration, ash dusting, certified virus-tested setts."
    },
    "Cotton": {
        "opt_rain": (600, 1100),
        "opt_temp": (21, 32),
        "soil": "Deep, fertile black cotton soils (vertisols) or well-drained alluvial loams.",
        "pests": "Bollworms (American, pink, spotted), aphids, jassids, whiteflies, bacterial blight.",
        "water": "Essential watering at square formation and boll development; avoid waterlogging.",
        "ipm": "Pheromone traps for pink bollworm, release of Trichogramma, neem oil for sucking pests."
    },
    "Sugarcane": {
        "opt_rain": (1200, 2000),
        "opt_temp": (24, 34),
        "soil": "Deep, well-drained loamy soils rich in organic matter and nutrients.",
        "pests": "Early shoot borer, top borer, red rot fungus, woolly aphid.",
        "water": "High water requirement; drip irrigation and trash mulching conserve moisture.",
        "ipm": "Sett treatment with carbendazim, biological control with egg parasitoids, rogue out red rot clumps."
    },
    "Barley": {
        "opt_rain": (400, 700),
        "opt_temp": (12, 22),
        "soil": "Well-drained loam or light clay; tolerates higher salinity and alkalinity than wheat.",
        "pests": "Aphids, armyworms, covered smut, loose smut, stripe rust.",
        "water": "Resilient in semi-arid zones; critical irrigation at tillering and boot stages.",
        "ipm": "Smut-resistant certified seed, vitavax seed dressing, synchronized regional sowing."
    },
    "Tomatoes": {
        "opt_rain": (600, 1000),
        "opt_temp": (18, 27),
        "soil": "Rich, well-drained sandy loam with pH 6.0-7.0; high organic humus content.",
        "pests": "Tomato fruit borer, whiteflies, early/late blight, leaf curl virus.",
        "water": "Consistent drip irrigation to avoid blossom-end rot and fruit cracking.",
        "ipm": "Yellow sticky cards for whiteflies, marigold border trap cropping, copper bactericides."
    },
    "Onions": {
        "opt_rain": (500, 800),
        "opt_temp": (15, 25),
        "soil": "Friable, fertile alluvial or sandy loam high in organic carbon (pH 6.5-7.5).",
        "pests": "Onion thrips, purple blotch fungus, basal rot, cutworms.",
        "water": "Frequent shallow irrigation; cease watering 10-15 days prior to bulb harvest.",
        "ipm": "Blue sticky traps for thrips, crop rotation with non-allium crops, seed/bulb dipping."
    },
    "Groundnuts": {
        "opt_rain": (500, 900),
        "opt_temp": (22, 30),
        "soil": "Light-textured loose sandy loam to encourage easy peg entry and pod development.",
        "pests": "Spodoptera litura, white grub, tikka leaf spot, collar rot.",
        "water": "Moderate moisture; critical during peg penetration and pod filling stages.",
        "ipm": "Trichoderma seed treatment, light traps for adult beetles, sulfur dust for leaf spot."
    },
    "Mustard & Rapeseed": {
        "opt_rain": (350, 650),
        "opt_temp": (12, 22),
        "soil": "Loam to heavy loam with adequate drainage; moderately drought and frost tolerant.",
        "pests": "Mustard aphid, painted bug, white rust, alternaria blight.",
        "water": "Low water demand; one or two irrigations at flowering and siliqua formation.",
        "ipm": "Yellow sticky traps, early sowing before aphid peaks, neem seed kernel extract (NSKE)."
    },
    "Chickpeas": {
        "opt_rain": (400, 700),
        "opt_temp": (15, 25),
        "soil": "Deep, well-drained silt loam to clay loam; sensitive to excessive wetness/salinity.",
        "pests": "Pod borer (Helicoverpa armigera), Fusarium wilt, collar rot.",
        "water": "Grown predominantly on residual soil moisture; excessive water causes vegetative rankness.",
        "ipm": "Rhizobium inoculation, bird perches for borer predation, pheromone monitoring traps."
    },
    "Bananas": {
        "opt_rain": (1500, 2500),
        "opt_temp": (24, 32),
        "soil": "Deep, fertile alluvial soil with high drainage capacity; sensitive to salt and drought.",
        "pests": "Pseudostem borer, banana aphid (bunchy top vector), panama wilt, sigatoka leaf spot.",
        "water": "Year-round plentiful moisture; mulching around stem base preserves moisture and prevents weed competition.",
        "ipm": "Tissue culture seedlings, injecting neem oil into infected pseudostems, sanitary de-suckering."
    }
}

def generate_farming_advice(country, crop, rainfall, temperature, pesticides, year, predicted_yield_hg_ha, historical_avg_hg_ha=None):
    """
    Generate farmer-friendly agronomic guidance and explanations.
    Uses Google Gemini API via official google-genai SDK if GEMINI_API_KEY is present.
    Falls back gracefully to a robust data-driven agronomic rule engine.
    """
    predicted_kg_ha = round(predicted_yield_hg_ha * 0.1, 2)
    predicted_tonnes_ha = round(predicted_yield_hg_ha * 0.0001, 3)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_response = None
    source = "Data-Driven Agronomic Intelligence"

    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            prompt = f"""
You are an expert Senior Agronomist and Agricultural Scientist assisting a farmer through AgriSense.
The machine learning model has already predicted the crop yield using verified historical FAO data.

--- FARMING CONTEXT ---
Location: {country}
Crop: {crop}
Production Year: {year}
Annual Rainfall: {rainfall} mm/year
Average Temperature: {temperature} °C
Pesticide Usage Level: {pesticides} tonnes (regional scale)
Model Predicted Yield: {predicted_yield_hg_ha:,.0f} hg/ha ({predicted_kg_ha:,.1f} kg/ha, {predicted_tonnes_ha:.2f} tonnes/ha)
Historical Dataset Average for {crop}: {historical_avg_hg_ha:,.0f} hg/ha if available else 'Standard Benchmark'

--- YOUR TASK ---
Provide practical, farmer-friendly, scientifically sound guidance tailored strictly to {crop} in {country}.
Structure your advice into these 6 exact numbered sections:
1. 🌱 Crop Suitability & Climate Fit: Evaluate if {temperature}°C and {rainfall} mm/year are optimal or stressful for {crop}.
2. 🌧 Rainfall & Water Management: Specific irrigation scheduling, drainage, or conservation tips for this rainfall level.
3. 🌡 Temperature & Weather Impact: Agronomic risks (heat stress, frost, humidity) and mitigation tactics.
4. 🐛 Integrated Pest Management (IPM): Responsible, low-chemical pest & disease control for major pests of {crop}.
5. 🌾 Practical Farming Recommendations: Sowing, soil preparation, balanced NPK fertilizing, and tillage advice.
6. 📈 Predicted Yield Interpretation: Plain-language explanation of what {predicted_kg_ha:,.1f} kg/ha means relative to expectations, and steps to protect or improve it.

Formatting: Use bullet points, bold key terms, clear language. Do NOT fabricate numerical yield values.
"""
            # Try gemini-2.5-flash
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            if response and response.text:
                gemini_response = response.text
                source = "Google Gemini AI (Agronomic Specialist)"
        except Exception as e:
            print(f"Gemini API invocation notice (using fallback): {e}")
            gemini_response = None

    if not gemini_response:
        gemini_response = _generate_rule_based_advice(
            country, crop, rainfall, temperature, pesticides, year,
            predicted_yield_hg_ha, predicted_kg_ha, predicted_tonnes_ha, historical_avg_hg_ha
        )

    return {
        "success": True,
        "source": source,
        "advice_text": gemini_response,
        "crop": crop,
        "country": country,
        "year": year,
        "predicted_yield_hg_ha": predicted_yield_hg_ha,
        "predicted_kg_ha": predicted_kg_ha,
        "predicted_tonnes_ha": predicted_tonnes_ha,
        "is_gemini": (source != "Data-Driven Agronomic Intelligence")
    }

def _generate_rule_based_advice(country, crop, rainfall, temperature, pesticides, year,
                                predicted_yield_hg_ha, predicted_kg_ha, predicted_tonnes_ha, historical_avg_hg_ha):
    """
    Comprehensive, scientific fallback agronomic guidance engine.
    Ensures that even without an external API key, the user gets rich, actionable advice.
    """
    profile = CROP_AGRONOMIC_PROFILES.get(crop)
    if not profile:
        try:
            from utils.crops_data import resolve_crop
            res = resolve_crop(crop)
            archetype = res.get("archetype")
            profile = CROP_AGRONOMIC_PROFILES.get(archetype)
        except Exception:
            profile = None
    if not profile:
        profile = {
            "opt_rain": (600, 1200),
            "opt_temp": (18, 28),
            "soil": "Fertile, well-drained loamy soil with balanced organic content.",
            "pests": "Common agricultural borers, fungal blights, and sap-sucking insects.",
            "water": "Maintain adequate root-zone moisture without prolonged standing water.",
            "ipm": "Regular field scouting, resistant seed selection, and targeted biological pest control."
        }

    rain_min, rain_max = profile["opt_rain"]
    temp_min, temp_max = profile["opt_temp"]

    # Rainfall assessment
    if rainfall < rain_min:
        rain_status = "Deficit / Sub-Optimal"
        rain_eval = f"At {rainfall} mm/year, annual precipitation is below the optimal requirement ({rain_min}–{rain_max} mm) for {crop}. Supplemental irrigation (e.g. drip or sprinkler systems) and mulch cover are strongly recommended to conserve soil moisture."
    elif rainfall > rain_max:
        rain_status = "Excess / High Moisture"
        rain_eval = f"At {rainfall} mm/year, rainfall exceeds normal thresholds ({rain_min}–{rain_max} mm). Raised seedbeds and adequate field drainage trenches are necessary to prevent waterlogging and root rot."
    else:
        rain_status = "Optimal Range"
        rain_eval = f"Rainfall of {rainfall} mm/year falls comfortably within the ideal agronomic range ({rain_min}–{rain_max} mm) for {crop}. Rainfall distribution across vegetative and reproductive growth phases should be tracked."

    # Temperature assessment
    if temperature < temp_min:
        temp_status = "Cool / Below Ideal"
        temp_eval = f"The recorded temperature ({temperature}°C) is cooler than the optimal thermal window ({temp_min}–{temp_max}°C). Germination and vegetative growth may be slow. Consider plastic mulching or adjusting sowing dates."
    elif temperature > temp_max:
        temp_status = "Elevated / Heat Stress"
        temp_eval = f"Average temperature of {temperature}°C exceeds optimal thresholds ({temp_min}–{temp_max}°C). High heat during flowering can lead to pollen sterility or rapid evapotranspiration. Frequent light irrigation and windbreaks help mitigate heat."
    else:
        temp_status = "Ideal Thermal Window"
        temp_eval = f"Average temperature of {temperature}°C matches the prime physiological range ({temp_min}–{temp_max}°C) for photosynthetic efficiency and yield formation in {crop}."

    # Yield comparison with historical average
    if historical_avg_hg_ha:
        hist_kg = historical_avg_hg_ha * 0.1
        diff_pct = ((predicted_kg_ha - hist_kg) / hist_kg) * 100
        if diff_pct > 5:
            yield_eval = f"The estimated yield of **{predicted_kg_ha:,.1f} kg/ha** is **{diff_pct:+.1f}% above** the historical regional baseline ({hist_kg:,.1f} kg/ha), indicating favorable environmental conditions and inputs."
        elif diff_pct < -5:
            yield_eval = f"The estimated yield of **{predicted_kg_ha:,.1f} kg/ha** is **{abs(diff_pct):.1f}% below** the historical average ({hist_kg:,.1f} kg/ha), suggesting moisture or thermal stress limiting potential productivity."
        else:
            yield_eval = f"The estimated yield of **{predicted_kg_ha:,.1f} kg/ha** closely aligns with the historical average ({hist_kg:,.1f} kg/ha)."
    else:
        yield_eval = f"The machine-learning pipeline estimates a yield of **{predicted_kg_ha:,.1f} kg/ha** ({predicted_tonnes_ha:.2f} tonnes/ha) under these conditions."

    # Pesticide assessment
    if pesticides > 25000:
        pest_eval = f"Reported regional pesticide volume ({pesticides:,.0f} tonnes) is high. Prioritize Integrated Pest Management (IPM), pheromone traps, and bio-pesticides to reduce chemical dependency, protect beneficial pollinators, and mitigate pesticide resistance."
    else:
        pest_eval = f"Pesticide intensity is within standard operational ranges. Follow recommended safety intervals and apply treatments only after field threshold scouting."

    sections = f"""
### 1. 🌱 Crop Suitability & Climate Fit
* **Location & Crop**: **{crop}** in **{country}** ({year}).
* **Thermal Fit**: {temp_status} ({temperature}°C vs optimal {temp_min}–{temp_max}°C).
* **Moisture Fit**: {rain_status} ({rainfall} mm/year vs optimal {rain_min}–{rain_max} mm).
* **Soil Requirements**: {profile['soil']}

### 2. 🌧 Rainfall & Water Management
* {rain_eval}
* **Water Strategy**: {profile['water']}
* Implement rainwater harvesting and furrow irrigation during dry spells.

### 3. 🌡 Temperature & Weather Impact
* {temp_eval}
* Maintain soil organic cover (mulch or green manure) to buffer root zone temperature swings.

### 4. 🐛 Integrated Pest Management (IPM)
* **Prevalent Pests & Pathogens**: {profile['pests']}.
* **Advisory**: {profile['ipm']}
* {pest_eval}

### 5. 🌾 Practical Farming Recommendations
* **Field Preparation**: Ensure proper land leveling and incorporate well-decomposed farmyard manure (FYM) or compost prior to planting.
* **Nutrient Management**: Conduct soil tests to calibrate balanced Nitrogen, Phosphorus, and Potassium (NPK) ratios. Split nitrogen applications to minimize leaching.
* **Crop Rotation**: Alternate {crop} with legumes (such as beans, peas, or clover) to replenish organic soil nitrogen and disrupt pest life cycles.

### 6. 📈 Predicted Yield Interpretation
* {yield_eval}
* **Expected Output**: **{predicted_yield_hg_ha:,.0f} hg/ha** = **{predicted_kg_ha:,.1f} kg/ha** = **{predicted_tonnes_ha:.2f} tonnes/ha**.
* *Note: Yield outcomes reflect mathematical estimations from historical trends and user-defined inputs. Actual field yields vary with soil microbiology, seed variety quality, and micro-climate patterns.*
"""
    return sections.strip()

def get_response(question):
    """
    Conversational fallback handler for general agricultural questions.
    """
    q = question.lower().strip()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = f"You are AgriSense AI, a helpful agricultural farming assistant. Answer this farmer's question concisely, accurately, and practically in 2-3 short paragraphs with bullet points if applicable: {question}"
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            if resp and resp.text:
                return resp.text
        except Exception as e:
            print(f"Gemini conversational notice: {e}")

    # Local conversational fallback
    qa_pairs = {
        "hello": "Hello! Welcome to the AgriSense Smart Farming Assistant. How can I help you with your crops, weather conditions, or yield predictions today?",
        "hi": "Hi there! Feel free to ask any question about crop management, soil health, irrigation, or pest control.",
        "crop": "Selecting the right crop depends on your country, rainfall, soil texture, and season. Use our Explore Crops feature to view historical yields for 10 major crops.",
        "rice": "Rice requires warm temperatures (22–32°C) and substantial water (1200–2000 mm/year). Alternate Wetting and Drying (AWD) is recommended to conserve water.",
        "wheat": "Wheat is a temperate cereal requiring cooler temperatures (14–22°C) and moderate rainfall (450–750 mm). Protect against rust diseases and avoid late heat stress.",
        "maize": "Maize prefers well-drained, fertile loamy soil with 600–1200 mm rainfall. Watch for fall armyworm and ensure sufficient moisture during the tasseling stage.",
        "potatoes": "Potatoes need loose, friable acidic soils (pH 5.2–6.5) and cool weather (15–20°C). Guard against late blight using certified disease-free seed tubers.",
        "cassava": "Cassava is extremely drought-resilient and thrives in sandy loam soils with 1000–1500 mm rainfall. Harvest roots within 9–12 months of planting.",
        "soybean": "Soybeans enrich soil through nitrogen fixation. Inoculate seeds with Bradyrhizobium and rotate with maize or wheat for optimal crop health.",
        "fertilizer": "Always base fertilizer applications on a recent soil test. Apply phosphorus and potassium at planting and split nitrogen applications to prevent leaching.",
        "pesticide": "Follow Integrated Pest Management (IPM). Scout fields regularly, use pheromone traps, promote beneficial predators, and spray chemical pesticides only as a last resort.",
        "irrigation": "Drip irrigation provides up to 90% water efficiency. Water crops early in the morning to minimize evaporative loss and fungal leaf infections.",
        "future": "Future-year yield predictions extrapolate patterns from the 1990–2013 historical training dataset. They are model-based estimates to assist in scenario planning."
    }

    for key, ans in qa_pairs.items():
        if key in q:
            return ans

    return (
        "I am here to help you optimize your farming! You can calibrate your farm conditions "
        "using the form above to get a comprehensive agronomic analysis for your specific crop and country, "
        "or ask questions regarding irrigation, soil preparation, pest management, and fertilizers."
    )