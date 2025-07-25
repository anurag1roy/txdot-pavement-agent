import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
import pandas as pd
import re

# 🔒 CONFIGURATION
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "transtecprojectsaa@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "fuvd qzkf tovs tomv")
TO_EMAILS = os.getenv("TO_EMAILS", "anurag@thetranstecgroup.com").split(",")
SHEET_LINK = f"https://docs.google.com/spreadsheets/d/{os.getenv('SPREADSHEET_ID', '18p3c-CbVHO7lc50V7fhnrqVUqgr2clkuyG8Y1oNhLbA')}/edit?usp=sharing"

date_str = datetime.now().strftime("%Y-%m-%d")
SUBJECT = f"Summary of New Projects from TxDOT Website - {date_str}"

# 🎯 COLUMNS FOR EMAIL TABLE - CLEANED UP
EMAIL_COLUMNS = [
    "Estimated Date of Advertisement",
    "Discipline of Procurement", 
    "District or Division Managing Contract",
    "Procuring PEPS Service Center",
    "Procurement Contact",
    "Selection Process",
    "Description of Procurement",
    "Project Location",
    "Contract Type",
    "Number of Contracts",
    "Estimated Maximum Contract Value",
    "Pavement Related?",
    "Summary",
    "Date Updated"
]

def send_email_with_table(new_rows_df, website_status=None):
    """
    📧 Send email with new projects table and website status report
    """
    
    # 🆕 Create website status section
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
            <p><small style="color: #e74c3c;">⚠️ Failed websites may affect completeness of results.</small></p>''' if error_sites else ""}
        </div>
        """
    
    if new_rows_df is not None and not new_rows_df.empty:
        # 🧹 Clean the data first
        df_clean = new_rows_df.copy()
        df_clean = df_clean.dropna(how='all')
        df_clean = df_clean.dropna(axis=1, how='all')
        
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
        
        # Create HTML table with proper headers
        html_parts = ['<table id="projects-table" class="dataframe">\n<thead>\n<tr>']
        
        # Add table headers
        for col in df_email.columns:
            html_parts.append(f'<th>{col}</th>')
        html_parts.append('</tr>\n</thead>\n<tbody>')
        
        # Add table rows with highlighting ONLY for high priority
        for idx, row in df_email.iterrows():
            # Check if this is a high priority row
            is_high_priority = 'High Priority' in str(row.get('Summary', ''))
            row_class = 'high-priority-row' if is_high_priority else ''
            
            html_parts.append(f'<tr class="{row_class}">')
            
            # Add each cell with proper formatting
            for col in df_email.columns:
                cell_value = str(row.get(col, ''))
                
                # Format the Summary column specifically
                if col == 'Summary' and '|' in cell_value:
                    parts = cell_value.split('|')
                    if len(parts) == 2:
                        business_line = parts[0].strip()
                        priority = parts[1].strip()
                        
                        formatted_value = f'<span class="business-line">{business_line}</span> | '
                        
                        if priority == 'High Priority':
                            formatted_value += f'<span class="priority-high">{priority}</span>'
                        elif priority == 'Medium Priority':
                            formatted_value += f'<span class="priority-medium">{priority}</span>'
                        elif priority == 'Low Priority':
                            formatted_value += f'<span class="priority-low">{priority}</span>'
                        else:
                            formatted_value += priority
                            
                        cell_value = formatted_value
                
                # Add the cell to the row
                html_parts.append(f'<td>{cell_value}</td>')
            
            html_parts.append('</tr>')
        
        html_parts.append('</tbody>\n</table>')
        html_table = '\n'.join(html_parts)
        
        # 💅 Enhanced CSS styling - SIMPLIFIED
        html_style = """
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0;
                padding: 20px;
                color: #333;
            }
            table.dataframe { 
                border-collapse: collapse; 
                width: 100%; 
                font-size: 12px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            table.dataframe th, table.dataframe td { 
                border: 1px solid #ddd; 
                padding: 8px; 
                text-align: left;
                vertical-align: top;
            }
            table.dataframe th { 
                background-color: #4CAF50; 
                color: white;
                font-weight: bold;
                text-align: center;
                position: sticky;
                top: 0;
            }
            table.dataframe tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            table.dataframe tr:hover {
                background-color: #f5f5f5;
            }
            /* ONLY High Priority Row Styling */
            .high-priority-row {
                background-color: #d4edda !important;
                border-left: 3px solid #28a745;
            }
            .high-priority-row:hover {
                background-color: #c3e6cb !important;
            }
            /* Business Line and Priority Styling */
            .business-line {
                font-weight: bold;
            }
            .priority-high {
                color: #155724;
                font-weight: bold;
            }
            .priority-medium {
                color: #856404;
                font-weight: bold;
            }
            .priority-low {
                color: #555;
            }
            /* Make description column wider */
            table.dataframe td:nth-child(7) {
                min-width: 250px;
            }
            h2, h3, h4 {
                margin-top: 20px;
                margin-bottom: 10px;
            }
        </style>
        """
        
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
            
            <div style="overflow-x: auto;">
                {html_table}
            </div>
            
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
            body {{ 
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
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

    # Clean up email list
    clean_to_emails = [email.strip() for email in TO_EMAILS if email.strip()]

    # 📤 Send the email
    msg = EmailMessage()
    msg["Subject"] = SUBJECT
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_ADDRESS
    msg["Bcc"] = ", ".join(clean_to_emails)
    
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

# 🧪 Test function
def test_email():
    """Test function to send a sample email with website status"""
    test_data = {
        "Estimated Date of Advertisement": ["2025-08-01", "2025-08-05", "2025-08-10"],
        "Discipline of Procurement": ["Engineering Services", "Material Testing", "Engineering Services"],
        "Description of Procurement": ["Pavement design for I-35", "Testing services for US-290", "Bridge inspection"],
        "Project Location": ["Austin", "Houston", "Dallas"],
        "Pavement Related?": ["Yes", "Yes", "No"],
        "Summary": ["Engineering|High Priority", "Testing|Medium Priority", "Engineering|Low Priority"],
        "Date Updated": ["2025-07-25", "2025-07-25", "2025-07-25"]
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
    test_email()