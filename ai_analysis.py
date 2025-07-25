import os
import pandas as pd
import logging
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe, set_with_dataframe

DESCRIPTION_COL = "Description of Procurement"
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

def is_pavement_related(description):
    desc_lower = str(description).lower()
    return any(keyword in desc_lower for keyword in PAVEMENT_KEYWORDS)

def analyze_and_update(df):
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
    
    for idx, row in df.iterrows():
        desc = str(row.get(DESCRIPTION_COL, ""))
        pavement = "Yes" if is_pavement_related(desc) else "No"
        df.at[idx, PAVEMENT_COL] = pavement
        df.at[idx, SUMMARY_COL] = "Keyword match" if pavement == "Yes" else ""
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
    # For testing: print new analyzed projects for today
    analyzed = analyze_new_today()
    if not analyzed.empty:
        print(analyzed)
    else:
        print("No new projects for today.")
    update_pavement_column_today_only()