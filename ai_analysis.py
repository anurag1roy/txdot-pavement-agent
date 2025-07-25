import os
import pandas as pd
import logging
import sys
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe, set_with_dataframe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import the ML classifier
try:
    from pavement_ml_classifier import TranstecMLClassifier
    ML_AVAILABLE = True
    logging.info("✅ ML classifier module loaded successfully")
except ImportError as e:
    logging.warning(f"⚠️ ML classifier not available ({e}). Will use keyword matching only.")
    ML_AVAILABLE = False

# Global classifier instance
classifier = None

DESCRIPTION_COL = "Description of Procurement"
PROJECT_TITLE_COL = "Project Title"
CLIENT_COL = "Client"
LOCATION_COL = "Project Location" 
PROJECT_VALUE_COL = "Estimated Maximum Contract Value"
PAVEMENT_COL = "Pavement Related?"
SUMMARY_COL = "Summary"
DATE_UPDATED_COL = "Date Updated"

PAVEMENT_KEYWORDS = [
    # Core pavement and materials
    "pavement", "asphalt", "concrete", "overlay", "resurfacing", "rehabilitation",
    "seal coat", "chip seal", "microsurfacing", "slurry seal", "mill and fill",
    "full-depth reclamation", "FDR", "flexible base", "subgrade", "hot mix",
    "HMA", "RAP", "recycled asphalt", "surface treatment", "roadway reconstruction",
    "roadway repair", "crack sealing", "joint sealing", "rut repair", "ride quality",
    "PCI", "pavement condition", "pavement design", "pavement evaluation",
    "pavement management", "pavement marking", "pavement testing", "core", "coring",
    "soil stabilization", "base stabilization", "shoulder repair", "shoulder widening",
    # Transtec/industry-specific and advanced services
    "materials testing", "construction management", "quality assurance", "QA/QC",
    "forensic engineering", "asset management", "life cycle cost analysis", "LCCA",
    "maintenance planning", "research", "data collection", "geotechnical", "structural engineering",
    "non-destructive testing", "NDT", "falling weight deflectometer", "FWD", "ground penetrating radar", "GPR",
    "automated data collection", "GIS mapping", "pavement condition survey", "performance modeling",
    "smart infrastructure", "innovative materials", "sustainable pavement", "green infrastructure",
    "recycled materials", "climate resilience", "low-carbon", "airport pavement", "runway", "taxiway",
    "highway", "interstate", "urban street", "municipal street", "public works", "DOT project",
    "statewide program", "citywide initiative", "international project", "federal project",
    # Maintenance and preservation
    "preventive maintenance", "preservation", "micro surfacing", "fog seal", "crack fill",
    "chip seal", "cape seal", "thin overlay", "ultra-thin overlay", "diamond grinding",
    "grooving", "patching", "full-depth repair", "partial-depth repair",
    # Airports and rail
    "airport", "runway", "taxiway", "apron", "railway", "track slab", "transit",
    # Miscellaneous
    "load transfer", "dowel bar", "tie bar", "joint sealant", "surface distress",
    "rutting", "faulting", "roughness", "skid resistance", "friction", "texture",
    "deflection", "modulus", "stiffness", "strength", "moisture damage", "thermal cracking",
    "reflection cracking", "block cracking", "longitudinal cracking", "transverse cracking",
    "raveling", "pothole", "edge drop", "shoulder drop", "base failure", "subgrade failure"
]

# Google Sheets config
SERVICE_ACCOUNT_FILENAME = os.getenv("SERVICE_ACCOUNT_FILE", "sage-passkey-465821-v5-5930e6f96f63.json")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "18p3c-CbVHO7lc50V7fhnrqVUqgr2clkuyG8Y1oNhLbA")
SHEET_NAME = "Summary"

def initialize_classifier():
    """Initialize the ML classifier if available"""
    global classifier, ML_AVAILABLE
    if not ML_AVAILABLE:
        return False
    
    # Try multiple potential model locations
    model_paths = [
        os.getenv("ML_MODEL_PATH"),  # First check environment variable
        "transtec_classifier_model.pkl",  # Then check local directory
        os.path.join(os.path.dirname(__file__), "transtec_classifier_model.pkl"),  # Check script directory
        "C:\\Users\\aanand\\OneDrive - Terracon Consultants Inc\\Desktop\\PythonProjects\\Agents\\transtec_classifier_model.pkl"  # Fallback to original path
    ]
    
    # Remove None values (from env var if not set)
    model_paths = [p for p in model_paths if p]
    
    for path in model_paths:
        try:
            logging.info(f"📁 Attempting to load model from: {path}")
            classifier = TranstecMLClassifier(model_path=path)
            logging.info(f"✅ ML classifier initialized with model from {path}")
            return True
        except Exception as e:
            logging.debug(f"Could not load model from {path}: {e}")
            continue
    
    # If we get here, all paths failed
    logging.error(f"❌ Failed to initialize ML classifier: Could not find model file")
    ML_AVAILABLE = False
    return False

def is_pavement_related(description):
    """Fallback keyword-based classification"""
    desc_lower = str(description).lower()
    return any(keyword in desc_lower for keyword in PAVEMENT_KEYWORDS)

def analyze_with_ml(project_data):
    """Analyze project using ML model with fallback to keyword matching"""
    global classifier, ML_AVAILABLE
    
    # Extract the description for fallback
    description = str(project_data.get('description', ''))
    project_name = str(project_data.get('project_name', ''))
    
    try:
        if not ML_AVAILABLE or classifier is None:
            # Fallback to keyword matching
            is_relevant = is_pavement_related(description)
            return is_relevant, "No" if not is_relevant else "Keyword", 0.0, "Keyword fallback"
            
        # Get ML prediction
        result = classifier.predict(project_data)
        
        # Format output - FIX: Properly check if it's not relevant
        business_line = result['predicted_class']
        confidence = result['confidence']
        recommendation = result['recommendation']
        
        # FIX: Correctly determine relevance
        is_relevant = False
        if business_line.lower() == "not_relevant" or business_line == 0:
            is_relevant = False
        else:
            # Check confidence threshold for relevance
            if confidence >= 0.40:  # Only mark as relevant if 40%+ confidence
                is_relevant = True
            else:
                # Fallback to keyword matching for low-confidence predictions
                is_relevant = is_pavement_related(description)
                if not is_relevant:
                    business_line = "No"
                    recommendation = "Low confidence - keyword verification failed"
        
        # Log the decision process
        logging.debug(f"Project '{project_name[:30]}': ML says '{business_line}' ({confidence:.1%}), Relevant: {is_relevant}")
        
        return is_relevant, business_line, confidence, recommendation
    except Exception as e:
        logging.error(f"ML prediction failed: {e}")
        # Fallback to keyword matching
        is_relevant = is_pavement_related(description)
        return is_relevant, "No" if not is_relevant else "Keyword", 0.0, "Error fallback to keywords"

def format_summary(business_line, confidence, recommendation):
    """Format a simplified summary with ML results"""
    if business_line == "No" or business_line == "Keyword":
        return "Keyword match" if business_line == "Keyword" else ""
        
    # Extract just the priority level from the recommendation
    priority = "Unknown"
    if "HIGH PRIORITY" in recommendation:
        priority = "High Priority"
    elif "MEDIUM PRIORITY" in recommendation:
        priority = "Medium Priority"
    elif "LOW PRIORITY" in recommendation:
        priority = "Low Priority"
        
    # Return simplified format
    return f"{business_line}|{priority}"

def analyze_and_update(df):
    """Analyze projects using ML and update dataframe"""
    # Initialize ML classifier if not already done
    ml_available = ML_AVAILABLE and (classifier is not None or initialize_classifier())
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure columns exist and are string type
    if PAVEMENT_COL not in df.columns:
        df[PAVEMENT_COL] = ""
    if SUMMARY_COL not in df.columns:
        df[SUMMARY_COL] = ""
    
    # Convert to string to avoid dtype warnings
    df[PAVEMENT_COL] = df[PAVEMENT_COL].astype(str)
    df[SUMMARY_COL] = df[SUMMARY_COL].astype(str)
    
    # Show progress for large datasets
    total_rows = len(df)
    logging.info(f"Analyzing {total_rows} projects...")
    
    for idx, row in df.iterrows():
        try:
            # Get available data
            desc = str(row.get(DESCRIPTION_COL, ""))
            
            if ml_available:
                # Prepare project data for ML
                project_data = {
                    'project_name': str(row.get(PROJECT_TITLE_COL, desc[:50])),
                    'description': desc,
                    'client': str(row.get(CLIENT_COL, "")),
                    'location': str(row.get(LOCATION_COL, "")),
                    'project_value': str(row.get(PROJECT_VALUE_COL, ""))
                }
                
                # Use ML analysis
                is_relevant, business_line, confidence, recommendation = analyze_with_ml(project_data)
                df.at[idx, PAVEMENT_COL] = "Yes" if is_relevant else "No"
                df.at[idx, SUMMARY_COL] = format_summary(business_line, confidence, recommendation)
                
                # Log progress periodically
                if idx % 20 == 0:
                    logging.info(f"Processed {idx} of {total_rows} projects...")
            else:
                # Fallback to keyword matching
                pavement = "Yes" if is_pavement_related(desc) else "No"
                df.at[idx, PAVEMENT_COL] = pavement
                df.at[idx, SUMMARY_COL] = "Keyword match" if pavement == "Yes" else ""
        except Exception as e:
            logging.error(f"Error analyzing row {idx}: {e}")
            # Don't modify existing values if there's an error
    
    logging.info(f"Analysis complete for {total_rows} projects")
    return df

def get_gsheet_df():
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILENAME, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SPREADSHEET_ID)
        worksheet = sh.worksheet(SHEET_NAME)
        df = get_as_dataframe(worksheet, evaluate_formulas=True, header=0).dropna(how="all")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        logging.error(f"Failed to load Google Sheet: {e}")
        return pd.DataFrame()

def analyze_new_today():
    df = get_gsheet_df()
    if df.empty:
        logging.info("No data found in master spreadsheet.")
        return pd.DataFrame()
    today_str = datetime.now().strftime("%Y-%m-%d")
    def extract_date(val):
        try:
            return pd.to_datetime(val).strftime("%Y-%m-%d")
        except Exception:
            return ""
    if DATE_UPDATED_COL not in df.columns:
        logging.error(f"❌ '{DATE_UPDATED_COL}' column not found in master spreadsheet.")
        return pd.DataFrame()
    mask = df[DATE_UPDATED_COL].apply(extract_date) == today_str
    new_rows = df[mask]
    if new_rows.empty:
        logging.info("No new projects found for today.")
        return pd.DataFrame()
    new_rows = analyze_and_update(new_rows)
    logging.info(f"Found {len(new_rows)} new projects for today.")
    return new_rows

def update_pavement_column_today_only():
    df = get_gsheet_df()
    if df.empty:
        logging.info("No data found in master spreadsheet.")
        return
    today_str = datetime.now().strftime("%Y-%m-%d")
    def extract_date(val):
        try:
            return pd.to_datetime(val).strftime("%Y-%m-%d")
        except Exception:
            return ""
    if DATE_UPDATED_COL not in df.columns:
        logging.error(f"❌ '{DATE_UPDATED_COL}' column not found in master spreadsheet.")
        return
    mask = df[DATE_UPDATED_COL].apply(extract_date) == today_str
    if not mask.any():
        logging.info("No rows found for today's date.")
        return
    # Update only today's rows, preserving index
    df[PAVEMENT_COL] = df[PAVEMENT_COL].astype(str)
    df[SUMMARY_COL] = df[SUMMARY_COL].astype(str)
    updated = analyze_and_update(df.loc[mask].copy())
    updated[PAVEMENT_COL] = updated[PAVEMENT_COL].astype(str)
    updated[SUMMARY_COL] = updated[SUMMARY_COL].astype(str)
    df.loc[mask, PAVEMENT_COL] = updated[PAVEMENT_COL]
    df.loc[mask, SUMMARY_COL] = updated[SUMMARY_COL]
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILENAME, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SPREADSHEET_ID)
        worksheet = sh.worksheet(SHEET_NAME)
        set_with_dataframe(worksheet, df, include_index=False, resize=True)
        logging.info("✅ Pavement Related? column updated for today's rows in Google Sheet.")
    except Exception as e:
        logging.error(f"Failed to update Google Sheet: {e}")

def update_pavement_column_all_rows():
    df = get_gsheet_df()
    if df.empty:
        logging.info("No data found in master spreadsheet.")
        return
    
    df = analyze_and_update(df)
    
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILENAME, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SPREADSHEET_ID)
        worksheet = sh.worksheet(SHEET_NAME)
        set_with_dataframe(worksheet, df, include_index=False, resize=True)
        logging.info("✅ Pavement Related? column updated for ALL rows in Google Sheet.")
    except Exception as e:
        logging.error(f"Failed to update Google Sheet: {e}")

if __name__ == "__main__":
    # Initialize the ML classifier
    if initialize_classifier():
        logging.info("🧠 Using ML classifier for project analysis")
    else:
        logging.warning("⚠️ Using keyword matching only (ML classifier not available)")
    
    # For testing: print new analyzed projects for today
    analyzed = analyze_new_today()
    if not analyzed.empty:
        print("\n--- Today's Analyzed Projects ---")
        print(analyzed[[PROJECT_TITLE_COL, DESCRIPTION_COL, PAVEMENT_COL, SUMMARY_COL]].head())
        print(f"Total: {len(analyzed)} projects")
    else:
        print("No new projects for today.")
    
    # Update today's rows in the spreadsheet
    update_pavement_column_today_only()