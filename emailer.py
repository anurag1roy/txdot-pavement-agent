import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
import pandas as pd

# 🔒 CONFIGURATION - Uses environment variables for security
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "transtecprojectsaa@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "fuvd qzkf tovs tomv")
TO_EMAILS = os.getenv("TO_EMAILS", "anurag@thetranstecgroup.com").split(",")
SHEET_LINK = f"https://docs.google.com/spreadsheets/d/{os.getenv('SPREADSHEET_ID', '18p3c-CbVHO7lc50V7fhnrqVUqgr2clkuyG8Y1oNhLbA')}/edit?usp=sharing"

date_str = datetime.now().strftime("%Y-%m-%d")
SUBJECT = f"Summary of New Projects from TxDOT Website - {date_str}"

# 🎯 CONFIGURABLE COLUMNS FOR EMAIL TABLE
# ✅ Uncomment any columns you WANT to include in the email
# ❌ Comment out any columns you DON'T want to include
EMAIL_COLUMNS = [
    "Estimated Date of Advertisement",
    "Discipline of Procurement", 
    "District or Division Managing Contract",
    "Procuring PEPS Service Center",
    "Procurement Contact",                    # 📧 Contact info
    # "Procurement Engineer (Do Not Contact)", # 🚫 Usually not needed
    "Selection Process",
    "Description of Procurement",             # 📝 Most important column
    "Project Location",
    "Contract Type",
    "Anticipated Funding Type",
    "Number of Contracts",
    "Estimated Maximum Contract Value",       # 💰 Important for business
    # "Pre-Solicitation Meeting",             # 📅 Usually not critical
    # "source_pdf",                           # 🔗 Technical info
    "Pavement Related?",                      # 🛣️ Key analysis result
    "Summary",                                # 📊 AI analysis summary
    # "Date added in list",                   # 📅 Internal tracking
    "Date Updated"                            # 📅 When it was processed
]

def send_email_with_table(new_rows_df):
    """
    📧 Send email with new projects table
    
    Args:
        new_rows_df: DataFrame with new projects (can be None or empty)
    """
    
    if new_rows_df is not None and not new_rows_df.empty:
        # 🧹 Clean the data first
        df_clean = new_rows_df.copy()
        df_clean = df_clean.dropna(how='all')  # Remove completely empty rows
        df_clean = df_clean.dropna(axis=1, how='all')  # Remove completely empty columns
        
        # 🎯 Select only the columns we want in the email
        available_columns = [col for col in EMAIL_COLUMNS if col in df_clean.columns]
        
        if available_columns:
            df_email = df_clean[available_columns]
            print(f"📧 Email will include these {len(available_columns)} columns:")
            for i, col in enumerate(available_columns, 1):
                print(f"   {i}. {col}")
        else:
            df_email = df_clean
            print(f"⚠️ No configured columns found, using all {len(df_clean.columns)} available columns")
        
        # 🎨 Create beautiful HTML table
        html_table = df_email.to_html(
            index=False, 
            border=0, 
            justify="center", 
            classes="dataframe", 
            escape=False,
            table_id="projects-table"
        )
        
        # 💅 Enhanced CSS styling
        html_style = """
        <style>
        body { font-family: Arial, sans-serif; }
        table.dataframe { 
            border-collapse: collapse; 
            width: 100%; 
            font-size: 11px;
            margin: 20px 0;
        }
        table.dataframe th, table.dataframe td { 
            border: 1px solid #ddd; 
            padding: 8px 10px; 
            text-align: left;
            vertical-align: top;
        }
        table.dataframe th { 
            background-color: #4CAF50; 
            color: white;
            font-weight: bold;
            text-align: center;
        }
        table.dataframe tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        table.dataframe tr:hover {
            background-color: #f5f5f5;
        }
        .pavement-yes {
            background-color: #d4edda !important;
            font-weight: bold;
            color: #155724;
        }
        .pavement-no {
            background-color: #f8d7da !important;
            color: #721c24;
        }
        </style>
        """
        
        # 🛣️ Highlight pavement-related projects
        if "Pavement Related?" in df_email.columns:
            html_table = html_table.replace(
                ">Yes<", 
                ' class="pavement-yes">✅ Yes<'
            ).replace(
                ">No<", 
                ' class="pavement-no">❌ No<'
            )
        
        # 📧 Create email body
        html_body = f"""
        <html>
        <head>{html_style}</head>
        <body>
        <h2>🛣️ TxDOT Pavement ProjectSeeker - Daily Report</h2>
        
        <p>Hello,</p>
        
        <p>📊 <strong>{len(df_email)} new projects</strong> have been found from the TxDOT projected contracts website.</p>
        
        <p>🔗 <a href="{SHEET_LINK}" target="_blank">View or edit the master spreadsheet here</a></p>
        
        {html_table}
        
        <hr>
        <p><small>
        📅 Generated on: {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}<br>
        🤖 Automated by: TxDOT Pavement ProjectSeeker<br>
        📋 Total columns shown: {len(available_columns) if available_columns else len(df_clean.columns)}
        </small></p>
        
        </body>
        </html>
        """
        
    else:
        # 📭 No new projects found
        html_body = f"""
        <html>
        <head>
        <style>
        body {{ font-family: Arial, sans-serif; }}
        .no-projects {{ 
            background-color: #fff3cd; 
            border: 1px solid #ffeaa7; 
            padding: 15px; 
            border-radius: 5px;
            margin: 20px 0;
        }}
        </style>
        </head>
        <body>
        <h2>🛣️ TxDOT Pavement ProjectSeeker - Daily Report</h2>
        
        <p>Hello,</p>
        
        <div class="no-projects">
        <p>📭 <strong>No new projects</strong> were found today.</p>
        <p>All projects from the TxDOT website are already in our database.</p>
        </div>
        
        <p>🔗 <a href="{SHEET_LINK}" target="_blank">View or edit the master spreadsheet here</a></p>
        
        <hr>
        <p><small>
        📅 Generated on: {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}<br>
        🤖 Automated by: TxDOT Pavement ProjectSeeker
        </small></p>
        
        </body>
        </html>
        """

    # 📤 Send the email
    msg = EmailMessage()
    msg["Subject"] = SUBJECT
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_ADDRESS  # Shows your email in "To" field
    msg["Bcc"] = ", ".join(TO_EMAILS)  # Actually sends to these people
    
    # Set both plain text and HTML versions
    msg.set_content("This email contains HTML content. Please view in an HTML-compatible email client.")
    msg.add_alternative(html_body, subtype="html")

    # 🚀 Send via SMTP
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"✅ Email sent successfully (BCC) to {TO_EMAILS}")
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        raise

# 🧪 Test function (for debugging)
def test_email():
    """Test function to send a sample email"""
    test_data = {
        "Description of Procurement": ["Test Project 1", "Test Project 2"],
        "Pavement Related?": ["Yes", "No"],
        "Project Location": ["Austin", "Houston"]
    }
    test_df = pd.DataFrame(test_data)
    send_email_with_table(test_df)

if __name__ == "__main__":
    # Run test if this file is executed directly
    test_email()