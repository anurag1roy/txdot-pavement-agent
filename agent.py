import extractor
import comparator
import ai_analysis
import emailer
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    try:
        logging.info("🚀 Starting TxDOT Pavement ProjectSeeker...")
        
        new_projects_df = extractor.extract_projects()
        # Try different ways to get website_status
        website_status = (
            getattr(new_projects_df, 'website_status', None) or  # Old way
            (new_projects_df.attrs.get('website_status', []) if hasattr(new_projects_df, 'attrs') else []) or  # New way
            []  # Fallback
        )
        
        if new_projects_df is not None and not new_projects_df.empty:
            logging.info(f"📊 Extracted {len(new_projects_df)} total projects from PDFs.")
            try:
                updated_master, new_rows = comparator.update_master_sheet(new_projects_df)
                
                if new_rows is not None and not new_rows.empty:
                    logging.info(f"✅ Found {len(new_rows)} NEW projects to add.")
                    new_rows = ai_analysis.analyze_and_update(new_rows)
                    logging.info(f"🔍 Pavement analysis completed for {len(new_rows)} new projects.")
                else:
                    logging.info("ℹ️ No new projects found (all projects already exist).")
                
                ai_analysis.update_pavement_column_today_only()
                
            except Exception as e:
                logging.error(f"Error updating master sheet or AI analysis: {e}")
                import traceback
                traceback.print_exc()
                new_rows = None
        else:
            logging.info("ℹ️ No projects extracted from PDFs.")
            new_rows = None
            
        try:
            # 🆕 Pass website status to emailer
            emailer.send_email_with_table(new_rows, website_status=website_status)
        except Exception as e:
            logging.error(f"Error sending email: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        logging.error(f"Fatal error in agent workflow: {e}")
        import traceback
        traceback.print_exc()