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
        if new_projects_df is not None and not new_projects_df.empty:
            logging.info(f"📊 Extracted {len(new_projects_df)} total projects from PDFs.")
            try:
                updated_master, new_rows = comparator.update_master_sheet(new_projects_df)
                
                if new_rows is not None and not new_rows.empty:
                    logging.info(f"✅ Found {len(new_rows)} NEW projects to add.")
                    # Analyze pavement for new rows BEFORE emailing
                    new_rows = ai_analysis.analyze_and_update(new_rows)
                    logging.info(f"🔍 Pavement analysis completed for {len(new_rows)} new projects.")
                else:
                    logging.info("ℹ️ No new projects found (all projects already exist).")
                
                # Update the entire master sheet with pavement analysis
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
            emailer.send_email_with_table(new_rows)
        except Exception as e:
            logging.error(f"Error sending email: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        logging.error(f"Fatal error in agent workflow: {e}")
        import traceback
        traceback.print_exc()