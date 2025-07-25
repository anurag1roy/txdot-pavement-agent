import os
import pdfplumber
import pandas as pd
import requests
from io import BytesIO
import logging
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import get_as_dataframe

# Google Sheets config
SERVICE_ACCOUNT_FILENAME = os.getenv("SERVICE_ACCOUNT_FILE", "sage-passkey-465821-v5-5930e6f96f63.json")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "18p3c-CbVHO7lc50V7fhnrqVUqgr2clkuyG8Y1oNhLbA")
WEBSITES_SHEET_NAME = "websites"
WEBSITES_CACHE_FILE = "websites_cache.csv"

def get_pdf_urls_from_sheet():
    urls = []
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILENAME, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SPREADSHEET_ID)
        print("Available worksheets:", [ws.title for ws in sh.worksheets()])
        worksheet = sh.worksheet(WEBSITES_SHEET_NAME)
        df = get_as_dataframe(worksheet, evaluate_formulas=True, header=0).dropna(how="all")
        urls = df["Website address"].dropna().astype(str).str.strip()
        urls = [url for url in urls if url.startswith("http")]
        # Save to cache
        pd.DataFrame({"Website address": urls}).to_csv(WEBSITES_CACHE_FILE, index=False)
        logging.info(f"Website URLs loaded from Google Sheet and cached to {WEBSITES_CACHE_FILE}")
    except Exception as e:
        logging.error(f"Failed to load website URLs from Google Sheet: {e}")
        # Try to load from cache
        if os.path.exists(WEBSITES_CACHE_FILE):
            try:
                cache_df = pd.read_csv(WEBSITES_CACHE_FILE)
                urls = cache_df["Website address"].dropna().astype(str).str.strip().tolist()
                logging.info(f"Website URLs loaded from local cache {WEBSITES_CACHE_FILE}")
            except Exception as cache_e:
                logging.error(f"Failed to load website URLs from cache: {cache_e}")
                urls = []
        else:
            logging.warning("No website cache file found.")
            urls = []
    return urls

def extract_projects():
    headers = [
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
        "Pre-Solicitation Meeting"
    ]
    pdf_urls = get_pdf_urls_from_sheet()
    logging.info(f"📄 Processing {len(pdf_urls)} PDF URLs from sheet.")
    
    all_tables = []
    website_status = []  # 🆕 Add this line
    
    for url in pdf_urls:
        website_name = url.split("/")[-1]
        projects_from_this_url = 0
        
        # 🆕 Add status tracking
        status_info = {
            "url": url,
            "name": website_name,
            "status": "unknown",
            "error": None,
            "projects_extracted": 0
        }
        
        try:
            logging.info(f"🌐 Opening website: {website_name}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            logging.info(f"✅ Website opened successfully: {website_name}")
            status_info["status"] = "success"  # 🆕 Mark as success
            
        except Exception as e:
            error_msg = f"HTTP error: {e}"
            logging.error(f"❌ Failed to open website {website_name}: {error_msg}")
            status_info["status"] = "error"  # 🆕 Mark as error
            status_info["error"] = error_msg
            website_status.append(status_info)  # 🆕 Add to status list
            continue
            
        try:
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                logging.info(f"📖 PDF opened, processing {len(pdf.pages)} pages from {website_name}")
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        tables = page.extract_tables()
                        for table_num, table in enumerate(tables, 1):
                            if table and len(table) > 1:
                                header_row = table[0]
                                if header_row != headers:
                                    logging.warning(
                                        f"Header mismatch on {website_name} page {page_num} table {table_num}. "
                                        f"Expected: {headers}, Found: {header_row}"
                                    )
                                data_rows = [row for row in table[1:] if len(row) == len(headers)]
                                data_rows = [row for row in data_rows if any(cell and str(cell).strip() for cell in row)]
                                if data_rows:
                                    df = pd.DataFrame(data_rows, columns=headers)
                                    df = df.loc[:, ~df.columns.duplicated()]
                                    df["source_pdf"] = website_name
                                    all_tables.append(df)
                                    projects_from_this_url += len(data_rows)
                    except Exception as e:
                        logging.error(f"Error extracting table from {website_name} page {page_num}: {e}")
            
            status_info["projects_extracted"] = projects_from_this_url
            logging.info(f"📊 Extracted {projects_from_this_url} projects from {website_name}")
            
        except Exception as e:
            error_msg = f"PDF processing error: {str(e)}"
            logging.error(f"❌ Error processing PDF from {website_name}: {error_msg}")
            status_info["status"] = "error"
            status_info["error"] = error_msg
        
        website_status.append(status_info)  # 🆕 Add status for this website
    
    # 🆕 Create final result with website status
    if all_tables:
        try:
            final_df = pd.concat(all_tables, ignore_index=True)
            final_df = final_df.drop_duplicates()
            final_df = final_df.dropna(how="all")
            logging.info(f"📊 TOTAL EXTRACTED: {len(final_df)} projects from all PDFs before comparison.")
        except Exception as e:
            logging.error(f"Error concatenating DataFrames: {e}")
            final_df = pd.DataFrame()
    else:
        logging.warning("⚠️ No valid tables found in any of the PDFs.")
        final_df = pd.DataFrame()
    
    # 🆕 At the very end, before returning:
    if hasattr(final_df, 'attrs'):
        final_df.attrs['website_status'] = website_status
    else:
        # Fallback: store as a separate variable and pass it along
        setattr(final_df, '_website_status', website_status)
    
    return final_df