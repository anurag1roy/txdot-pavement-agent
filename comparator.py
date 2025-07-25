import os
import pandas as pd
import gspread
import logging
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe, set_with_dataframe

# 🔒 CONFIGURATION - Uses environment variables for security  
SERVICE_ACCOUNT_FILENAME = os.getenv("SERVICE_ACCOUNT_FILE", "sage-passkey-465821-v5-5930e6f96f63.json")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "18p3c-CbVHO7lc50V7fhnrqVUqgr2clkuyG8Y1oNhLbA")
SHEET_NAME = "Summary"
DESCRIPTION_COL = "Description of Procurement"
DATE_UPDATED_COL = "Date Updated"

# Update the SHEET_HEADERS to match your actual data structure
SHEET_HEADERS = [
    "Estimated Date of Advertisement",
    "Discipline of Procurement", 
    "District or Division Managing Contract",
    "Procuring PEPS Service Center",
    "Procurement Contact",
    "Procurement Engineer (Do Not Contact)",
    "Selection Process", 
    "Description of Procurement",
    "Project Location",
    "Contract Type",
    "Anticipated Funding Type",
    "Number of Contracts",
    "Estimated Maximum Contract Value",
    "Pre-Solicitation Meeting",
    "source_pdf",
    "Pavement Related?",
    "Summary",
    "Date added in list", 
    "Date Updated"
]

def get_gsheet():
    try:
        print("Looking for credentials at:", SERVICE_ACCOUNT_FILENAME)  # ← Fixed this line
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILENAME, scopes=scopes)  # ← And this line
        print("Service account email:", creds.service_account_email)
        gc = gspread.authorize(creds)
        print("gspread authorized.")
        sh = gc.open_by_key(SPREADSHEET_ID)
        print("Spreadsheet opened.")
        print("Available worksheets:", [ws.title for ws in sh.worksheets()])
        worksheet = sh.worksheet(SHEET_NAME)
        print("Worksheet opened.")
        return worksheet
    except Exception as e:
        import traceback
        logging.error(f"Failed to connect to Google Sheet: {e}")
        traceback.print_exc()
        raise

def update_master_sheet(new_df):
    try:
        worksheet = get_gsheet()
        master_df = get_as_dataframe(worksheet, evaluate_formulas=True, header=0).dropna(how="all")
        master_df.columns = master_df.columns.str.strip()

        # Ensure all headers are present in master_df, add missing ones
        for col in SHEET_HEADERS:
            if col not in master_df.columns:
                master_df[col] = ""

        # For new_df, only add columns that don't exist but are needed
        for col in SHEET_HEADERS:
            if col not in new_df.columns:
                if col in ["Pavement Related?", "Summary", "Date added in list", "Date Updated"]:
                    new_df[col] = ""  # These will be filled later

        # Reorder columns to match SHEET_HEADERS (only for columns that exist)
        master_cols = [col for col in SHEET_HEADERS if col in master_df.columns]
        new_cols = [col for col in SHEET_HEADERS if col in new_df.columns]
        
        master_df = master_df[master_cols]
        new_df = new_df[new_cols]

        # Remove accidental header rows from new_df
        new_df = new_df[new_df[DESCRIPTION_COL].str.strip().str.lower() != DESCRIPTION_COL.strip().lower()]

        # Normalize descriptions for robust comparison
        def normalize_desc(s):
            return str(s).replace('\n', ' ').replace('\r', ' ').strip().lower()

        master_descriptions = set(master_df[DESCRIPTION_COL].apply(normalize_desc))
        new_rows_mask = ~new_df[DESCRIPTION_COL].apply(normalize_desc).isin(master_descriptions)
        new_rows = new_df[new_rows_mask]

        if new_rows.empty:
            logging.info("✅ No new unique rows to add.")
            return master_df, pd.DataFrame()

        now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        new_rows = new_rows.copy()  # Add this line first
        new_rows[DATE_UPDATED_COL] = now_str
        
        # Ensure new_rows has all columns that master_df has
        for col in master_df.columns:
            if col not in new_rows.columns:
                new_rows[col] = ""

        # Reorder new_rows to match master_df columns
        new_rows = new_rows[master_df.columns]

        updated_master = pd.concat([master_df, new_rows], ignore_index=True)
        set_with_dataframe(worksheet, updated_master, include_index=False, resize=True)
        logging.info(f"✅ Added {len(new_rows)} new unique rows to master spreadsheet.")
        return updated_master, new_rows

    except Exception as e:
        logging.error(f"Error updating master sheet: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()