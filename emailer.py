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

def send_email_with_table(new_rows_df, website_status=None):
    """
    📧 Send email with new projects table and website status report
    
    Args:
        new_rows_df: DataFrame with new projects (can be None or empty)
        website_status: List of website status info (success and failures)
    """
    
    # 🆕 Create website status section (shows ALL websites checked)
    website_status_section = ""
    if website_status:
        success_sites = [site for site in website_status if site["status"] == "success"]
        error_sites = [site for site in website_status if site["status"] == "error"]
        
        total_projects = sum(site.get("projects_extracted", 0) for site in success_sites)
        
        # Success section
        success_details = []
        for site in success_sites:
            projects_count = site.get("projects_extracted", 0)
            success_details.append(f"• <strong>{site['name']}</strong>: ✅ Success - {projects_count} projects extracted")
        
        # Error section  
        error_details = []
        for site in error_sites:
            error_details.append(f"• <strong>{site['name']}</strong>: ❌ {site['error']}")
        
        website_status_section = f"""
        <div style="background-color: #e9f7ef; border: 1px solid #27ae60; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h3 style="color: #27ae60; margin-top: 0;">🌐 Website Status Report</h3>
        <p><strong>Websites Checked: {len(website_status)} | Successful: {len(success_sites)} | Failed: {len(error_sites)}</strong></p>
        
        {f'''<h4 style="color: #27ae60; margin: 10px 0 5px 0;">✅ Successful Websites ({len(success_sites)}):</h4>
        <ul style="margin: 5px 0 15px 20px;">
        {"".join(f"<li>{detail}</li>" for detail in success_details)}
        </ul>''' if success_sites else ""}
        
        {f'''<h4 style="color: #e74c3c; margin: 10px 0 5px 0;">❌ Failed Websites ({len(error_sites)}):</h4>
        <ul style="margin: 5px 0 15px 20px;">
        {"".join(f"<li>{detail}</li>" for detail in error_details)}
        </ul>
        <p><small style="color: #e74c3c;">⚠️ Failed websites may affect completeness of results. Please check manually if needed.</small></p>''' if error_sites else ""}
        </div>
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
        
        # 📧 Create email body with website status section
        html_body = f"""
        <html>
        <head>{html_style}</head>
        <body>
        <h2>🛣️ TxDOT Pavement ProjectSeeker - Daily Report</h2>
        
        <p>Hello,</p>
        
        <p>📊 <strong>{len(df_email)} new projects</strong> have been found from the TxDOT projected contracts website.</p>
        
        <p>🔗 <a href="{SHEET_LINK}" target="_blank">View or edit the master spreadsheet here</a></p>
        
        {website_status_section}
        
        {html_table}
        
        <hr>
        <p><small>
        📅 Generated on: {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}<br>
        🤖 Automated by: TxDOT Pavement ProjectSeeker<br>
        📋 Total columns shown: {len(available_columns) if available_columns else len(df_clean.columns)}
        {f"<br>🌐 Websites checked: {len(website_status)} | ✅ Success: {len([s for s in website_status if s['status'] == 'success'])} | ❌ Failed: {len([s for s in website_status if s['status'] == 'error'])}" if website_status else ""}
        </small></p>
        
        </body>
        </html>
        """
        
    else:
        # 📭 No new projects found - but still show website status
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
        
        {website_status_section}
        
        <hr>
        <p><small>
        📅 Generated on: {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}<br>
        🤖 Automated by: TxDOT Pavement ProjectSeeker
        {f"<br>🌐 Websites checked: {len(website_status)} | ✅ Success: {len([s for s in website_status if s['status'] == 'success'])} | ❌ Failed: {len([s for s in website_status if s['status'] == 'error'])}" if website_status else ""}
        </small></p>
        
        </body>
        </html>
        """

    # Clean up email list (remove empty strings and whitespace)
    clean_to_emails = [email.strip() for email in TO_EMAILS if email.strip()]

    # 📤 Send the email
    msg = EmailMessage()
    msg["Subject"] = SUBJECT
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_ADDRESS  # Shows your email in "To" field
    msg["Bcc"] = ", ".join(clean_to_emails)  # Actually sends to these people
    
    # Set both plain text and HTML versions
    msg.set_content("This email contains HTML content. Please view in an HTML-compatible email client.")
    msg.add_alternative(html_body, subtype="html")

    # 🚀 Send via SMTP
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"✅ Email sent successfully (BCC) to {clean_to_emails}")
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        raise

# 🧪 Test function (for debugging)
def test_email():
    """Test function to send a sample email with website status"""
    test_data = {
        "Description of Procurement": ["Test Project 1", "Test Project 2"],
        "Pavement Related?": ["Yes", "No"],
        "Project Location": ["Austin", "Houston"],
        "Date Updated": ["2025-07-25", "2025-07-25"]
    }
    test_df = pd.DataFrame(test_data)
    
    # Test website status
    test_website_status = [
        {
            "url": "https://example.com/good.pdf",
            "name": "good.pdf",
            "status": "success",
            "projects_extracted": 2
        },
        {
            "url": "https://example.com/bad.pdf",
            "name": "bad.pdf", 
            "status": "error",
            "error": "404 Not Found"
        }
    ]
    
    send_email_with_table(test_df, website_status=test_website_status)

if __name__ == "__main__":
    # Run test if this file is executed directly
    test_email()