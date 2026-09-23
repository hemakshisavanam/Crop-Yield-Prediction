"""
AgriSense - Crop Data & Analytics Utility
Loads metadata and cached dataset statistics for fast, real-data lookups.
"""

import os
import json
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_METADATA_PATH = os.path.join(_BASE_DIR, "model", "metadata.json")
_DATASET_PATH = os.path.join(_BASE_DIR, "dataset", "yield_df.csv")

_cached_metadata = None
_cached_df = None

def get_metadata():
    global _cached_metadata
    if _cached_metadata is None:
        if os.path.exists(_METADATA_PATH):
            with open(_METADATA_PATH, "r", encoding="utf-8") as f:
                _cached_metadata = json.load(f)
        else:
            _cached_metadata = {}
    return _cached_metadata

def get_clean_dataframe():
    global _cached_df
    if _cached_df is None:
        if os.path.exists(_DATASET_PATH):
            df = pd.read_csv(_DATASET_PATH)
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            _cached_df = df.drop_duplicates()
        else:
            _cached_df = pd.DataFrame()
    return _cached_df

# Core FAO dataset crops (model training baseline)
CORE_CROPS = [
    "Cassava", "Maize", "Plantains and others", "Potatoes",
    "Rice, paddy", "Sorghum", "Soybeans", "Sweet potatoes",
    "Wheat", "Yams"
]

# Agronomic categories and extensive catalog of 40+ agricultural crops
EXTENDED_CROPS_CATALOG = {
    # Cereals & Grains
    "Maize": {
        "category": "Cereals & Grains",
        "archetype": "Maize",
        "scaling_factor": 1.0,
        "mean_yield_kg_ha": 3032.0,
        "aliases": ["corn", "sweet corn", "makka"],
        "icon": "fa-seedling"
    },
    "Rice, paddy": {
        "category": "Cereals & Grains",
        "archetype": "Rice, paddy",
        "scaling_factor": 1.0,
        "mean_yield_kg_ha": 4002.0,
        "aliases": ["rice", "paddy", "chawal", "dhan"],
        "icon": "fa-wheat-awn"
    },
    "Wheat": {
        "category": "Cereals & Grains",
        "archetype": "Wheat",
        "scaling_factor": 1.0,
        "mean_yield_kg_ha": 3020.0,
        "aliases": ["wheat grain", "gehun", "durum"],
        "icon": "fa-wheat-awn"
    },
    "Sorghum": {
        "category": "Cereals & Grains",
        "archetype": "Sorghum",
        "scaling_factor": 1.0,
        "mean_yield_kg_ha": 1390.0,
        "aliases": ["jowar", "milo", "great millet"],
        "icon": "fa-wheat-awn"
    },
    "Barley": {
        "category": "Cereals & Grains",
        "archetype": "Wheat",
        "scaling_factor": 1.05,
        "mean_yield_kg_ha": 3170.0,
        "aliases": ["barley grain", "jau", "hordeum"],
        "icon": "fa-wheat-awn"
    },
    "Millets": {
        "category": "Cereals & Grains",
        "archetype": "Sorghum",
        "scaling_factor": 0.95,
        "mean_yield_kg_ha": 1320.0,
        "aliases": ["bajra", "ragi", "pearl millet", "finger millet", "foxtail millet"],
        "icon": "fa-wheat-awn"
    },
    "Oats": {
        "category": "Cereals & Grains",
        "archetype": "Wheat",
        "scaling_factor": 0.85,
        "mean_yield_kg_ha": 2560.0,
        "aliases": ["oat", "jai", "avena"],
        "icon": "fa-wheat-awn"
    },
    "Rye": {
        "category": "Cereals & Grains",
        "archetype": "Wheat",
        "scaling_factor": 0.90,
        "mean_yield_kg_ha": 2710.0,
        "aliases": ["rye grain", "secale"],
        "icon": "fa-wheat-awn"
    },

    # Pulses & Legumes
    "Soybeans": {
        "category": "Pulses & Legumes",
        "archetype": "Soybeans",
        "scaling_factor": 1.0,
        "mean_yield_kg_ha": 1720.0,
        "aliases": ["soybean", "soya", "soy", "bhatmash"],
        "icon": "fa-seedling"
    },
    "Chickpeas": {
        "category": "Pulses & Legumes",
        "archetype": "Soybeans",
        "scaling_factor": 0.85,
        "mean_yield_kg_ha": 1460.0,
        "aliases": ["chickpea", "gram", "chana", "garbanzo", "bengal gram"],
        "icon": "fa-seedling"
    },
    "Lentils": {
        "category": "Pulses & Legumes",
        "archetype": "Soybeans",
        "scaling_factor": 0.80,
        "mean_yield_kg_ha": 1370.0,
        "aliases": ["lentil", "masoor", "dhal", "red lentil"],
        "icon": "fa-seedling"
    },
    "Groundnuts": {
        "category": "Pulses & Legumes",
        "archetype": "Soybeans",
        "scaling_factor": 1.10,
        "mean_yield_kg_ha": 1890.0,
        "aliases": ["peanut", "peanuts", "groundnut", "moongphali", "monkey nut"],
        "icon": "fa-seedling"
    },
    "Peas": {
        "category": "Pulses & Legumes",
        "archetype": "Soybeans",
        "scaling_factor": 1.15,
        "mean_yield_kg_ha": 1980.0,
        "aliases": ["green peas", "dry peas", "matar", "field pea"],
        "icon": "fa-seedling"
    },
    "Pigeon Peas": {
        "category": "Pulses & Legumes",
        "archetype": "Soybeans",
        "scaling_factor": 0.75,
        "mean_yield_kg_ha": 1290.0,
        "aliases": ["arhar", "tur", "red gram", "toor dal"],
        "icon": "fa-seedling"
    },
    "Black Gram": {
        "category": "Pulses & Legumes",
        "archetype": "Soybeans",
        "scaling_factor": 0.65,
        "mean_yield_kg_ha": 1120.0,
        "aliases": ["urad", "urad dal", "mash", "black lentil"],
        "icon": "fa-seedling"
    },
    "Green Gram": {
        "category": "Pulses & Legumes",
        "archetype": "Soybeans",
        "scaling_factor": 0.65,
        "mean_yield_kg_ha": 1120.0,
        "aliases": ["moong", "mung", "mung bean", "green moong"],
        "icon": "fa-seedling"
    },

    # Commercial & Cash Crops
    "Sugarcane": {
        "category": "Cash & Commercial",
        "archetype": "Cassava",
        "scaling_factor": 5.5,
        "mean_yield_kg_ha": 72500.0,
        "aliases": ["sugar cane", "cane", "ganna"],
        "icon": "fa-plant-wilt"
    },
    "Cotton": {
        "category": "Cash & Commercial",
        "archetype": "Soybeans",
        "scaling_factor": 1.15,
        "mean_yield_kg_ha": 1980.0,
        "aliases": ["raw cotton", "cotton lint", "kapas"],
        "icon": "fa-shirt"
    },
    "Coffee": {
        "category": "Cash & Commercial",
        "archetype": "Plantains and others",
        "scaling_factor": 0.15,
        "mean_yield_kg_ha": 1280.0,
        "aliases": ["coffee bean", "arabica", "robusta"],
        "icon": "fa-mug-hot"
    },
    "Tea": {
        "category": "Cash & Commercial",
        "archetype": "Plantains and others",
        "scaling_factor": 0.25,
        "mean_yield_kg_ha": 2130.0,
        "aliases": ["tea leaf", "camellia", "chai"],
        "icon": "fa-leaf"
    },
    "Tobacco": {
        "category": "Cash & Commercial",
        "archetype": "Soybeans",
        "scaling_factor": 1.10,
        "mean_yield_kg_ha": 1890.0,
        "aliases": ["tobacco leaf", "nicotiana"],
        "icon": "fa-smoking"
    },
    "Jute": {
        "category": "Cash & Commercial",
        "archetype": "Maize",
        "scaling_factor": 0.80,
        "mean_yield_kg_ha": 2420.0,
        "aliases": ["jute fiber", "pat", "golden fiber"],
        "icon": "fa-lines-leaning"
    },
    "Rubber": {
        "category": "Cash & Commercial",
        "archetype": "Plantains and others",
        "scaling_factor": 0.18,
        "mean_yield_kg_ha": 1530.0,
        "aliases": ["natural rubber", "latex", "hevea"],
        "icon": "fa-circle-dot"
    },

    # Vegetables & Horticulture
    "Tomatoes": {
        "category": "Vegetables & Horticulture",
        "archetype": "Potatoes",
        "scaling_factor": 1.10,
        "mean_yield_kg_ha": 34100.0,
        "aliases": ["tomato", "tamatar", "lycopersicon"],
        "icon": "fa-apple-whole"
    },
    "Onions": {
        "category": "Vegetables & Horticulture",
        "archetype": "Potatoes",
        "scaling_factor": 0.65,
        "mean_yield_kg_ha": 20150.0,
        "aliases": ["onion", "pyaz", "shallot", "shallots"],
        "icon": "fa-circle"
    },
    "Garlic": {
        "category": "Vegetables & Horticulture",
        "archetype": "Potatoes",
        "scaling_factor": 0.35,
        "mean_yield_kg_ha": 10850.0,
        "aliases": ["lehsun", "garlic bulb"],
        "icon": "fa-clover"
    },
    "Cabbage": {
        "category": "Vegetables & Horticulture",
        "archetype": "Potatoes",
        "scaling_factor": 0.90,
        "mean_yield_kg_ha": 27900.0,
        "aliases": ["patta gobhi", "head cabbage", "brassica"],
        "icon": "fa-circle-notch"
    },
    "Cauliflower": {
        "category": "Vegetables & Horticulture",
        "archetype": "Potatoes",
        "scaling_factor": 0.60,
        "mean_yield_kg_ha": 18600.0,
        "aliases": ["phool gobhi", "broccoli", "cauliflower head"],
        "icon": "fa-asterisk"
    },
    "Carrots": {
        "category": "Vegetables & Horticulture",
        "archetype": "Potatoes",
        "scaling_factor": 0.85,
        "mean_yield_kg_ha": 26350.0,
        "aliases": ["carrot", "gajar", "daucus"],
        "icon": "fa-carrot"
    },
    "Eggplant": {
        "category": "Vegetables & Horticulture",
        "archetype": "Potatoes",
        "scaling_factor": 0.70,
        "mean_yield_kg_ha": 21700.0,
        "aliases": ["brinjal", "aubergine", "baingan", "egg plant"],
        "icon": "fa-egg"
    },
    "Chili Peppers": {
        "category": "Vegetables & Horticulture",
        "archetype": "Potatoes",
        "scaling_factor": 0.20,
        "mean_yield_kg_ha": 6200.0,
        "aliases": ["chili", "chilli", "mirch", "peppers", "capsicum", "bell pepper", "hot pepper"],
        "icon": "fa-pepper-hot"
    },
    "Okra": {
        "category": "Vegetables & Horticulture",
        "archetype": "Soybeans",
        "scaling_factor": 4.5,
        "mean_yield_kg_ha": 7740.0,
        "aliases": ["bhindi", "ladyfinger", "lady finger", "gumbo"],
        "icon": "fa-hand-point-right"
    },
    "Spinach": {
        "category": "Vegetables & Horticulture",
        "archetype": "Potatoes",
        "scaling_factor": 0.45,
        "mean_yield_kg_ha": 13950.0,
        "aliases": ["palak", "spinach greens", "spinacia"],
        "icon": "fa-leaf"
    },

    # Fruits & Plantation
    "Plantains and others": {
        "category": "Fruits & Plantation",
        "archetype": "Plantains and others",
        "scaling_factor": 1.0,
        "mean_yield_kg_ha": 8500.0,
        "aliases": ["plantains", "cooking banana"],
        "icon": "fa-lemon"
    },
    "Bananas": {
        "category": "Fruits & Plantation",
        "archetype": "Plantains and others",
        "scaling_factor": 2.2,
        "mean_yield_kg_ha": 18700.0,
        "aliases": ["banana", "kela", "musa"],
        "icon": "fa-lemon"
    },
    "Apples": {
        "category": "Fruits & Plantation",
        "archetype": "Plantains and others",
        "scaling_factor": 1.8,
        "mean_yield_kg_ha": 15300.0,
        "aliases": ["apple", "seb", "malus"],
        "icon": "fa-apple-whole"
    },
    "Mangoes": {
        "category": "Fruits & Plantation",
        "archetype": "Plantains and others",
        "scaling_factor": 1.1,
        "mean_yield_kg_ha": 9350.0,
        "aliases": ["mango", "aam", "mangifera"],
        "icon": "fa-lemon"
    },
    "Grapes": {
        "category": "Fruits & Plantation",
        "archetype": "Plantains and others",
        "scaling_factor": 1.3,
        "mean_yield_kg_ha": 11050.0,
        "aliases": ["grape", "angur", "vineyard", "vitis"],
        "icon": "fa-wine-glass"
    },
    "Papayas": {
        "category": "Fruits & Plantation",
        "archetype": "Plantains and others",
        "scaling_factor": 3.5,
        "mean_yield_kg_ha": 29750.0,
        "aliases": ["papaya", "papita", "carica"],
        "icon": "fa-lemon"
    },
    "Watermelons": {
        "category": "Fruits & Plantation",
        "archetype": "Potatoes",
        "scaling_factor": 0.95,
        "mean_yield_kg_ha": 29450.0,
        "aliases": ["watermelon", "tarbooz", "citrullus", "melon"],
        "icon": "fa-circle"
    },
    "Citrus / Oranges": {
        "category": "Fruits & Plantation",
        "archetype": "Plantains and others",
        "scaling_factor": 1.7,
        "mean_yield_kg_ha": 14450.0,
        "aliases": ["orange", "citrus", "santra", "lemon", "lime", "kinnow"],
        "icon": "fa-lemon"
    },

    # Oilseeds
    "Mustard & Rapeseed": {
        "category": "Oilseeds",
        "archetype": "Soybeans",
        "scaling_factor": 0.90,
        "mean_yield_kg_ha": 1548.0,
        "aliases": ["mustard", "rapeseed", "sarson", "canola", "raya"],
        "icon": "fa-droplet"
    },
    "Sunflower": {
        "category": "Oilseeds",
        "archetype": "Soybeans",
        "scaling_factor": 1.05,
        "mean_yield_kg_ha": 1806.0,
        "aliases": ["sunflower seed", "surajmukhi", "helianthus"],
        "icon": "fa-sun"
    },
    "Sesame": {
        "category": "Oilseeds",
        "archetype": "Sorghum",
        "scaling_factor": 0.45,
        "mean_yield_kg_ha": 625.0,
        "aliases": ["sesame seed", "til", "gingelly"],
        "icon": "fa-droplet"
    },

    # Roots & Tubers
    "Potatoes": {
        "category": "Roots & Tubers",
        "archetype": "Potatoes",
        "scaling_factor": 1.0,
        "mean_yield_kg_ha": 31000.0,
        "aliases": ["potato", "aloo", "spud"],
        "icon": "fa-cubes"
    },
    "Sweet potatoes": {
        "category": "Roots & Tubers",
        "archetype": "Sweet potatoes",
        "scaling_factor": 1.0,
        "mean_yield_kg_ha": 13900.0,
        "aliases": ["sweet potato", "shakarkand", "batata"],
        "icon": "fa-cubes"
    },
    "Cassava": {
        "category": "Roots & Tubers",
        "archetype": "Cassava",
        "scaling_factor": 1.0,
        "mean_yield_kg_ha": 13100.0,
        "aliases": ["manioc", "yuca", "tapioca"],
        "icon": "fa-cubes"
    },
    "Yams": {
        "category": "Roots & Tubers",
        "archetype": "Yams",
        "scaling_factor": 1.0,
        "mean_yield_kg_ha": 10200.0,
        "aliases": ["yam", "jimikand", "suran"],
        "icon": "fa-cubes"
    }
}

# Category defaults for custom / manual crops
CATEGORY_ARCHETYPE_DEFAULTS = {
    "Cereals & Grains": {"archetype": "Wheat", "scaling_factor": 1.0, "mean_yield_kg_ha": 3000.0},
    "Pulses & Legumes": {"archetype": "Soybeans", "scaling_factor": 0.9, "mean_yield_kg_ha": 1550.0},
    "Cash & Commercial": {"archetype": "Soybeans", "scaling_factor": 1.1, "mean_yield_kg_ha": 1900.0},
    "Vegetables & Horticulture": {"archetype": "Potatoes", "scaling_factor": 0.75, "mean_yield_kg_ha": 23000.0},
    "Fruits & Plantation": {"archetype": "Plantains and others", "scaling_factor": 1.5, "mean_yield_kg_ha": 12750.0},
    "Oilseeds": {"archetype": "Soybeans", "scaling_factor": 0.9, "mean_yield_kg_ha": 1550.0},
    "Roots & Tubers": {"archetype": "Potatoes", "scaling_factor": 0.8, "mean_yield_kg_ha": 24800.0},
    "General / Other": {"archetype": "Maize", "scaling_factor": 1.0, "mean_yield_kg_ha": 3000.0}
}

def get_all_crops():
    """Returns the 10 core FAO historical training crops."""
    metadata = get_metadata()
    return metadata.get("crops", CORE_CROPS)

def get_all_selectable_crops():
    """Returns all 40+ crops available for selection (alphabetical)."""
    return sorted(list(EXTENDED_CROPS_CATALOG.keys()))

def get_crop_categories():
    """
    Returns structured categories with icons and crop lists for UI grouping.
    """
    categories = [
        {"name": "Cereals & Grains", "icon": "fa-wheat-awn", "badge": "Cereals"},
        {"name": "Pulses & Legumes", "icon": "fa-seedling", "badge": "Pulses"},
        {"name": "Cash & Commercial", "icon": "fa-sack-dollar", "badge": "Commercial"},
        {"name": "Vegetables & Horticulture", "icon": "fa-carrot", "badge": "Vegetables"},
        {"name": "Fruits & Plantation", "icon": "fa-apple-whole", "badge": "Fruits"},
        {"name": "Oilseeds", "icon": "fa-droplet", "badge": "Oilseeds"},
        {"name": "Roots & Tubers", "icon": "fa-cubes", "badge": "Tubers"}
    ]
    for cat in categories:
        cat["crops"] = [
            crop for crop, info in EXTENDED_CROPS_CATALOG.items()
            if info["category"] == cat["name"]
        ]
    return categories

def resolve_crop(crop_name, manual_category=None):
    """
    Resolves any crop name (core, catalog, alias, or manual user input)
    into a structured resolution with model archetype, scaling factor, and benchmark stats.
    """
    if not crop_name:
        crop_name = "Maize"

    clean_name = str(crop_name).strip()
    clean_lower = clean_name.lower()
    core_crops = get_all_crops()

    # 1. Exact Core Crop match
    for c in core_crops:
        if c.lower() == clean_lower:
            meta = get_metadata()
            stats = meta.get("crop_statistics", {}).get(c, {})
            cat = EXTENDED_CROPS_CATALOG.get(c, {}).get("category", "General")
            return {
                "crop": c,
                "is_core": True,
                "is_extended": False,
                "is_manual": False,
                "archetype": c,
                "scaling_factor": 1.0,
                "category": cat,
                "mean_yield_kg_ha": stats.get("mean_yield_kg_ha", 3000.0),
                "source_label": "Direct FAO Benchmark Model"
            }

    # 2. Exact Extended Crop match
    for c, info in EXTENDED_CROPS_CATALOG.items():
        if c.lower() == clean_lower:
            return {
                "crop": c,
                "is_core": False,
                "is_extended": True,
                "is_manual": False,
                "archetype": info["archetype"],
                "scaling_factor": info["scaling_factor"],
                "category": info["category"],
                "mean_yield_kg_ha": info["mean_yield_kg_ha"],
                "source_label": f"Calibrated via FAO {info['archetype']} Model"
            }

    # 3. Alias match in Extended Catalog
    for c, info in EXTENDED_CROPS_CATALOG.items():
        for alias in info.get("aliases", []):
            if alias.lower() == clean_lower:
                return {
                    "crop": c,
                    "is_core": False,
                    "is_extended": True,
                    "is_manual": False,
                    "archetype": info["archetype"],
                    "scaling_factor": info["scaling_factor"],
                    "category": info["category"],
                    "mean_yield_kg_ha": info["mean_yield_kg_ha"],
                    "source_label": f"Calibrated via FAO {info['archetype']} Model"
                }

    # 4. Keyword heuristic matching for manual crops
    inferred_category = manual_category or "General / Other"
    if not manual_category or manual_category not in CATEGORY_ARCHETYPE_DEFAULTS:
        if any(w in clean_lower for w in ["grain", "cereal", "wheat", "corn", "millet", "paddy", "oat", "barley", "rye"]):
            inferred_category = "Cereals & Grains"
        elif any(w in clean_lower for w in ["pulse", "bean", "pea", "lentil", "gram", "dal", "legume"]):
            inferred_category = "Pulses & Legumes"
        elif any(w in clean_lower for w in ["cane", "sugar", "cotton", "tobacco", "coffee", "tea", "jute", "rubber"]):
            inferred_category = "Cash & Commercial"
        elif any(w in clean_lower for w in ["tomato", "onion", "garlic", "cabbage", "carrot", "pepper", "chili", "veg", "greens", "leaf", "brinjal"]):
            inferred_category = "Vegetables & Horticulture"
        elif any(w in clean_lower for w in ["fruit", "apple", "banana", "mango", "berry", "grape", "orange", "citrus", "melon", "papaya"]):
            inferred_category = "Fruits & Plantation"
        elif any(w in clean_lower for w in ["oil", "mustard", "sunflower", "sesame", "canola"]):
            inferred_category = "Oilseeds"
        elif any(w in clean_lower for w in ["root", "tuber", "potato", "yam", "cassava", "radish", "turnip", "beet"]):
            inferred_category = "Roots & Tubers"

    default_info = CATEGORY_ARCHETYPE_DEFAULTS.get(inferred_category, CATEGORY_ARCHETYPE_DEFAULTS["General / Other"])

    # Properly capitalized display name for the manual crop
    display_name = clean_name.title() if clean_name.islower() else clean_name

    return {
        "crop": display_name,
        "is_core": False,
        "is_extended": False,
        "is_manual": True,
        "archetype": default_info["archetype"],
        "scaling_factor": default_info["scaling_factor"],
        "category": inferred_category,
        "mean_yield_kg_ha": default_info["mean_yield_kg_ha"],
        "source_label": f"Manual Custom Crop (Calibrated via {default_info['archetype']})"
    }

def get_all_areas():
    metadata = get_metadata()
    return metadata.get("areas", [])

def search_crops(query):
    """
    Search crops dynamically across both core and extended catalogs,
    supporting exact, prefix, substring, and alias matching.
    """
    if not query:
        return {
            "match_found": False,
            "results": get_all_selectable_crops(),
            "best_match": None,
            "query": "",
            "can_enter_manual": True
        }

    q = query.strip().lower()
    selectable_crops = get_all_selectable_crops()

    # 1. Exact match
    for crop in selectable_crops:
        if crop.lower() == q:
            return {
                "match_found": True,
                "results": [crop],
                "best_match": crop,
                "query": query,
                "can_enter_manual": False
            }

    # 2. Prefix match
    prefix_matches = [crop for crop in selectable_crops if crop.lower().startswith(q)]
    if prefix_matches:
        return {
            "match_found": True,
            "results": prefix_matches,
            "best_match": prefix_matches[0],
            "query": query,
            "can_enter_manual": True
        }

    # 3. Substring match
    substring_matches = [crop for crop in selectable_crops if q in crop.lower()]

    # 4. Check aliases if substring matches are sparse
    alias_matches = []
    for crop, info in EXTENDED_CROPS_CATALOG.items():
        if crop not in substring_matches:
            for alias in info.get("aliases", []):
                if q in alias.lower():
                    alias_matches.append(crop)
                    break

    combined = list(dict.fromkeys(substring_matches + alias_matches))
    if combined:
        return {
            "match_found": True,
            "results": combined,
            "best_match": combined[0],
            "query": query,
            "can_enter_manual": True
        }

    return {
        "match_found": False,
        "results": [],
        "best_match": None,
        "query": query,
        "can_enter_manual": True
    }

def get_crop_details(crop_name):
    """
    Get detailed statistics and yearly trend for a specific crop.
    Supports core FAO crops as well as extended catalog crops.
    """
    metadata = get_metadata()
    crop_stats = metadata.get("crop_statistics", {}).get(crop_name)
    if not crop_stats:
        # Try case-insensitive lookup
        for k, v in metadata.get("crop_statistics", {}).items():
            if k.lower() == crop_name.lower():
                crop_name = k
                crop_stats = v
                break

    if not crop_stats:
        # Check if it's an extended crop
        resolution = resolve_crop(crop_name)
        if resolution["is_extended"] or resolution["is_manual"]:
            archetype = resolution["archetype"]
            arch_stats = metadata.get("crop_statistics", {}).get(archetype, {})
            factor = resolution["scaling_factor"]
            mean_hg = round(arch_stats.get("mean_yield_hg_ha", 30000.0) * factor, 1)
            mean_kg = round(mean_hg * 0.1, 1)

            df = get_clean_dataframe()
            arch_df = df[df["Item"] == archetype] if not df.empty else pd.DataFrame()

            yearly_trend = []
            if not arch_df.empty:
                trend = arch_df.groupby("Year")["hg/ha_yield"].mean().round(2).reset_index()
                yearly_trend = [
                    {"year": int(row["Year"]), "yield_hg_ha": round(float(row["hg/ha_yield"]) * factor, 1), "yield_kg_ha": round(float(row["hg/ha_yield"]) * factor * 0.1, 1)}
                    for _, row in trend.iterrows()
                ]

            top_areas = []
            if not arch_df.empty:
                areas_df = arch_df.groupby("Area")["hg/ha_yield"].agg(["count", "mean"]).reset_index()
                areas_df = areas_df.sort_values(by="mean", ascending=False).head(10)
                top_areas = [
                    {
                        "area": str(row["Area"]),
                        "records": int(row["count"]),
                        "avg_yield_hg_ha": round(float(row["mean"]) * factor, 1),
                        "avg_yield_kg_ha": round(float(row["mean"]) * factor * 0.1, 1)
                    }
                    for _, row in areas_df.iterrows()
                ]

            return {
                "crop": resolution["crop"],
                "category": resolution["category"],
                "is_extended": resolution["is_extended"],
                "is_manual": resolution["is_manual"],
                "archetype": archetype,
                "scaling_factor": factor,
                "statistics": {
                    "records": arch_stats.get("records", 1000),
                    "countries_count": arch_stats.get("countries_count", 50),
                    "mean_yield_hg_ha": mean_hg,
                    "mean_yield_kg_ha": mean_kg,
                    "mean_yield_tonnes_ha": round(mean_kg / 1000.0, 3),
                    "min_yield_hg_ha": round(arch_stats.get("min_yield_hg_ha", 5000.0) * factor, 1),
                    "max_yield_hg_ha": round(arch_stats.get("max_yield_hg_ha", 60000.0) * factor, 1),
                    "std_yield_hg_ha": round(arch_stats.get("std_yield_hg_ha", 5000.0) * factor, 1),
                    "avg_rainfall_mm": arch_stats.get("avg_rainfall_mm", 1000.0),
                    "avg_temp_c": arch_stats.get("avg_temp_c", 20.0)
                },
                "yearly_trend": yearly_trend,
                "top_areas": top_areas,
                "available_areas": sorted(arch_df["Area"].unique().tolist()) if not arch_df.empty else []
            }
        return None

    df = get_clean_dataframe()
    crop_df = df[df["Item"] == crop_name] if not df.empty else pd.DataFrame()

    # Yearly trend
    yearly_trend = []
    if not crop_df.empty:
        trend = crop_df.groupby("Year")["hg/ha_yield"].mean().round(2).reset_index()
        yearly_trend = [
            {"year": int(row["Year"]), "yield_hg_ha": float(row["hg/ha_yield"]), "yield_kg_ha": round(float(row["hg/ha_yield"]) * 0.1, 2)}
            for _, row in trend.iterrows()
        ]

    # Top countries/areas for this crop
    top_areas = []
    if not crop_df.empty:
        areas_df = crop_df.groupby("Area")["hg/ha_yield"].agg(["count", "mean"]).reset_index()
        areas_df = areas_df.sort_values(by="mean", ascending=False).head(10)
        top_areas = [
            {
                "area": str(row["Area"]),
                "records": int(row["count"]),
                "avg_yield_hg_ha": round(float(row["mean"]), 2),
                "avg_yield_kg_ha": round(float(row["mean"]) * 0.1, 2)
            }
            for _, row in areas_df.iterrows()
        ]

    return {
        "crop": crop_name,
        "statistics": crop_stats,
        "yearly_trend": yearly_trend,
        "top_areas": top_areas,
        "available_areas": sorted(crop_df["Area"].unique().tolist()) if not crop_df.empty else []
    }

def get_dashboard_chart_data():
    """
    Compute real dataset aggregations for the 8 dashboard charts.
    """
    df = get_clean_dataframe()
    metadata = get_metadata()

    if df.empty:
        return {}

    # 1. Average Yield by Crop (hg/ha and kg/ha)
    crop_avg = df.groupby("Item")["hg/ha_yield"].mean().sort_values(ascending=False)
    chart1 = {
        "labels": crop_avg.index.tolist(),
        "yield_hg_ha": [round(float(val), 2) for val in crop_avg.values],
        "yield_kg_ha": [round(float(val) * 0.1, 2) for val in crop_avg.values]
    }

    # 2. Historical Yield Trend by Year (1990-2013)
    year_avg = df.groupby("Year")["hg/ha_yield"].mean().sort_index()
    chart2 = {
        "labels": [str(y) for y in year_avg.index],
        "yield_kg_ha": [round(float(val) * 0.1, 2) for val in year_avg.values]
    }

    # 3. Feature Importance
    feat_imp = metadata.get("feature_importances", {})
    readable_map = {
        "Area": "Country / Region",
        "Item": "Crop Variety",
        "Year": "Production Year",
        "average_rain_fall_mm_per_year": "Annual Rainfall",
        "pesticides_tonnes": "Pesticide Usage",
        "avg_temp": "Average Temperature"
    }
    chart3 = {
        "labels": [readable_map.get(k, k) for k in feat_imp.keys()],
        "values": [round(float(v) * 100, 2) for v in feat_imp.values()]
    }

    # 4. Top 10 Countries by Mean Crop Yield
    top_countries = df.groupby("Area")["hg/ha_yield"].mean().sort_values(ascending=False).head(10)
    chart4 = {
        "labels": top_countries.index.tolist(),
        "yield_kg_ha": [round(float(val) * 0.1, 2) for val in top_countries.values]
    }

    # 5. Rainfall Bins vs Mean Yield
    df["rainfall_bin"] = pd.cut(
        df["average_rain_fall_mm_per_year"],
        bins=[0, 500, 1000, 1500, 2000, 3500],
        labels=["<500mm", "500-1000mm", "1000-1500mm", "1500-2000mm", ">2000mm"]
    )
    rain_vs_yield = df.groupby("rainfall_bin", observed=False)["hg/ha_yield"].mean()
    chart5 = {
        "labels": [str(b) for b in rain_vs_yield.index],
        "yield_kg_ha": [round(float(val) * 0.1, 2) for val in rain_vs_yield.values]
    }

    # 6. Temperature Bins vs Mean Yield
    df["temp_bin"] = pd.cut(
        df["avg_temp"],
        bins=[-5, 10, 15, 20, 25, 35],
        labels=["<10°C", "10-15°C", "15-20°C", "20-25°C", ">25°C"]
    )
    temp_vs_yield = df.groupby("temp_bin", observed=False)["hg/ha_yield"].mean()
    chart6 = {
        "labels": [str(b) for b in temp_vs_yield.index],
        "yield_kg_ha": [round(float(val) * 0.1, 2) for val in temp_vs_yield.values]
    }

    # 7. Pesticide Usage vs Mean Yield
    df["pest_bin"] = pd.cut(
        df["pesticides_tonnes"],
        bins=[-1, 500, 5000, 20000, 100000, 400000],
        labels=["<500t", "500-5kt", "5k-20kt", "20k-100kt", ">100kt"]
    )
    pest_vs_yield = df.groupby("pest_bin", observed=False)["hg/ha_yield"].mean()
    chart7 = {
        "labels": [str(b) for b in pest_vs_yield.index],
        "yield_kg_ha": [round(float(val) * 0.1, 2) for val in pest_vs_yield.values]
    }

    # 8. Crop Min vs Avg vs Max Yield (kg/ha)
    crop_stats = metadata.get("crop_statistics", {})
    chart8 = {
        "labels": list(crop_stats.keys()),
        "min_kg_ha": [round(s["min_yield_hg_ha"] * 0.1, 1) for s in crop_stats.values()],
        "mean_kg_ha": [round(s["mean_yield_hg_ha"] * 0.1, 1) for s in crop_stats.values()],
        "max_kg_ha": [round(s["max_yield_hg_ha"] * 0.1, 1) for s in crop_stats.values()]
    }

    return {
        "chart1_crop_yield": chart1,
        "chart2_yearly_trend": chart2,
        "chart3_feature_importance": chart3,
        "chart4_top_countries": chart4,
        "chart5_rainfall_vs_yield": chart5,
        "chart6_temp_vs_yield": chart6,
        "chart7_pest_vs_yield": chart7,
        "chart8_yield_spread": chart8
    }
