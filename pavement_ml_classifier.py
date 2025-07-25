import pandas as pd
import numpy as np
import pickle
import re
import os
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime

warnings.filterwarnings('ignore')

# Import ML libraries with error handling
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import StandardScaler
    import joblib
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("📦 Install with: pip install scikit-learn pandas numpy")
    exit(1)

class TranstecMLClassifier:
    """
    🧠 Machine Learning classifier for Transtec Group project relevance
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.vectorizer = None
        self.feature_scaler = None
        self.is_trained = False
        self.model_path = model_path or "transtec_classifier_model.pkl"
        self.feedback_log = []
        self.training_history = []
        
        # Business line categories for your 4 focus areas
        self.business_lines = {
            0: "not_relevant",
            1: "Research",
            2: "Engineering", 
            3: "Testing",
            4: "Tracks"
        }
        
        self.setup_feature_extractors()

    def setup_feature_extractors(self):
        """🔧 Initialize feature extraction components"""
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            strip_accents='ascii',
            min_df=1,
            max_df=0.95
        )
        
        self.feature_scaler = StandardScaler()
        
        # Keyword categories for each business line
        self.keyword_categories = {
            'research_keywords': [
                'research', 'development', 'innovative', 'experimental', 'pilot',
                'study', 'investigation', 'analysis', 'performance', 'advanced', 
                'technology', 'hiperpav', 'proval', 'optimization', 'evaluation'
            ],
            'engineering_keywords': [
                'design', 'engineering', 'structural', 'pavement design', 'concrete design',
                'asphalt design', 'specifications', 'standards', 'calculation', 'modeling',
                'cad', 'drawings', 'plans', 'layout', 'geometry', 'rehabilitation'
            ],
            'testing_keywords': [
                'testing', 'evaluation', 'assessment', 'inspection', 'analysis',
                'fwd', 'falling weight deflectometer', 'gpr', 'ground penetrating radar',
                'condition', 'structural testing', 'non-destructive', 'ndt', 'survey'
            ],
            'tracks_keywords': [
                'track', 'racetrack', 'racing', 'motorsports', 'speedway', 'circuit',
                'raceway', 'automotive', 'performance track', 'test track',
                'high speed', 'banking', 'surface properties', 'grip', 'safety'
            ],
            'pavement_core': [
                'pavement', 'asphalt', 'concrete', 'roadway', 'highway', 'runway',
                'surface', 'base', 'subgrade', 'construction', 'materials'
            ]
        }

    def load_training_data_from_excel(self, excel_path: str) -> pd.DataFrame:
        """📊 Load and prepare training data from Excel file"""
        print(f"📁 Loading training data from: {excel_path}")
        
        # Check if file exists
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel file not found at {excel_path}")
            
        try:
            # Try to load with pandas - with specific error handling
            try:
                df = pd.read_excel(excel_path)
            except PermissionError:
                print("⚠️ Permission denied. The file might be open in another program.")
                print("   Please close any programs that might be using the file and try again.")
                raise
            except Exception as e:
                print(f"⚠️ Error reading Excel file: {str(e)}")
                raise
                
            print(f"✅ Loaded {len(df)} rows from Excel")
            print(f"📋 Original columns: {list(df.columns)}")
            
            # Map columns to expected format
            column_mapping = {
                'Project Title': 'project_name',
                'Line of Business': 'business_line_text',
                'Client': 'client',
                'Total Contract Amount': 'project_value',
                'Project Description': 'description'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Map business line text to numeric labels (ignore PSC)
            business_line_mapping = {
                'Research': 1,
                'Engineering': 2,
                'Testing': 3,
                'Tracks': 4,
                'Not Relevant': 0,
                'not_relevant': 0,
                'research': 1,
                'engineering': 2,
                'testing': 3,
                'tracks': 4
            }
            
            # Create label column based on business line text
            if 'business_line_text' in df.columns:
                df['label'] = df['business_line_text'].map(business_line_mapping)
                
                # Show unmapped labels for debugging
                unmapped = df[df['label'].isna()]['business_line_text'].unique()
                if len(unmapped) > 0:
                    print(f"⚠️ Unmapped business lines found: {unmapped} - Setting to not_relevant")
                    # Set unmapped (including PSC) to not_relevant
                    df['label'] = df['label'].fillna(0)
            else:
                print("⚠️ No business line column found. You'll need to add labels manually.")
                df['label'] = 0  # Default to not_relevant
            
            # Clean and validate required columns
            required_columns = ['project_name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns after mapping: {missing_columns}")
            
            print(f"✅ Data mapping complete. Columns: {list(df.columns)}")
            print(f"📊 Business line distribution:")
            label_counts = df['label'].value_counts()
            for label, count in label_counts.items():
                business_line = self.business_lines.get(int(label), 'unknown')
                print(f"   {business_line}: {count} samples")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            raise

    def validate_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """✅ Validate and clean training data - keep all data with project names"""
        print("🔍 Validating training data...")
        
        if df is None:
            raise ValueError("Training data cannot be None")
        
        if df.empty:
            raise ValueError("Training data cannot be empty")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Check required columns
        required_columns = ['project_name', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean data
        text_columns = ['project_name', 'description', 'location', 'client', 'project_value']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        
        # Remove only rows with empty project names
        initial_count = len(df)
        df = df[df['project_name'].str.strip() != '']
        removed_empty_names = initial_count - len(df)
        
        if removed_empty_names > 0:
            print(f"⚠️ Removed {removed_empty_names} rows with empty project names")
        
        if df.empty:
            raise ValueError("No valid data remains after removing rows with empty project names")
        
        # Count missing descriptions but keep all rows
        if 'description' in df.columns:
            missing_desc_count = df['description'].str.strip().eq('').sum()
            if missing_desc_count > 0:
                print(f"📋 {missing_desc_count} rows have missing descriptions - will use project name and other data")
        
        # Validate labels
        valid_labels = list(self.business_lines.keys())
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        invalid_mask = ~df['label'].isin(valid_labels) | df['label'].isna()
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            print(f"⚠️ Found {invalid_count} rows with invalid labels. Removing...")
            df = df[~invalid_mask]
        
        if df.empty:
            raise ValueError("No valid data remains after removing rows with invalid labels")
        
        # Show class distribution
        class_counts = df['label'].value_counts()
        print(f"📊 Class distribution:")
        for label, count in class_counts.items():
            business_line = self.business_lines.get(int(label), 'unknown')
            print(f"   {business_line}: {count} samples")
        
        # Check if we have enough data for training
        if len(df) < 4:
            raise ValueError(f"Need at least 4 samples for training (found {len(df)})")
        
        # Check if we have at least 2 classes
        if len(class_counts) < 2:
            raise ValueError(f"Need at least 2 different classes (found {len(class_counts)})")
        
        print(f"✅ Validation complete. {len(df)} valid samples ready for training.")
        return df

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """🔍 Extract features that work with available data"""
        features_list = []
        
        for index, row in df.iterrows():
            try:
                # Get available text data
                project_name = str(row.get('project_name', ''))
                description = str(row.get('description', ''))
                location = str(row.get('location', ''))
                client = str(row.get('client', ''))
                project_value = str(row.get('project_value', ''))
                
                # Build text content with smart weighting
                if description.strip() == '' or description.lower() in ['nan', 'none']:
                    # No description - focus on project name
                    text_content = f"{project_name} " * 5  # 5x weight for project name
                    text_content += f"{client} {location} {project_value}"
                else:
                    # Description available - balanced weighting
                    text_content = f"{project_name} {project_name} "  # 2x weight
                    text_content += f"{description} {description} {description} "  # 3x weight
                    text_content += f"{client} {location} {project_value}"
                
                # Calculate keyword density for each category
                keyword_features = []
                for category, keywords in self.keyword_categories.items():
                    if keywords:
                        # Count matches with weighting
                        name_matches = sum(1 for keyword in keywords if keyword in project_name.lower())
                        desc_matches = sum(1 for keyword in keywords if keyword in description.lower())
                        other_matches = sum(1 for keyword in keywords if keyword in f"{client} {location}".lower())
                        
                        # Weighted scoring
                        weighted_score = (name_matches * 3) + (desc_matches * 2) + other_matches
                        max_possible = len(keywords) * 6
                        density = min(weighted_score / max_possible, 1.0) if max_possible > 0 else 0.0
                    else:
                        density = 0.0
                    keyword_features.append(density)
                
                # Simple numerical features
                numerical_features = [
                    self.extract_project_value(text_content),
                    len(project_name.split()),
                    len(description.split()) if description.strip() else 0,
                    len(text_content.split()),
                    self.get_client_score(client),
                    self.get_location_score(location)
                ]
                
                # Combine all features
                combined_features = keyword_features + numerical_features
                combined_features = [float(f) if not np.isnan(float(f)) and not np.isinf(float(f)) else 0.0 for f in combined_features]
                features_list.append(combined_features)
                
            except Exception as e:
                print(f"⚠️ Error processing row {index}: {e}")
                # Add zero features for failed rows
                num_features = len(self.keyword_categories) + 6
                features_list.append([0.0] * num_features)
        
        return np.array(features_list, dtype=np.float32)

    def extract_text_features(self, df: pd.DataFrame) -> np.ndarray:
        """📝 Extract TF-IDF features from text content"""
        text_data = []
        
        for index, row in df.iterrows():
            try:
                project_name = str(row.get('project_name', ''))
                description = str(row.get('description', ''))
                location = str(row.get('location', ''))
                client = str(row.get('client', ''))
                
                text_content = f"{project_name} {description} {location} {client}"
                text_data.append(text_content)
            except:
                text_data.append("")
        
        # Handle empty text data
        if not text_data or all(text == "" for text in text_data):
            if self.is_trained and hasattr(self.vectorizer, 'get_feature_names_out'):
                # Return zero array with correct dimensions
                return np.zeros((len(text_data), len(self.vectorizer.get_feature_names_out())))
            else:
                raise ValueError("No text data available for feature extraction")
                
        if self.is_trained:
            return self.vectorizer.transform(text_data).toarray()
        else:
            return self.vectorizer.fit_transform(text_data).toarray()

    def extract_project_value(self, text: str) -> float:
        """💰 Extract and normalize project value"""
        try:
            text = str(text).lower()
            value_patterns = [
                (r'\$([0-9,]+\.?[0-9]*)\s*(billion|b)', 1000000000),
                (r'\$([0-9,]+\.?[0-9]*)\s*(million|m)', 1000000),
                (r'\$([0-9,]+\.?[0-9]*)\s*k', 1000),
                (r'\$([0-9,]+)', 1)
            ]
            
            for pattern, multiplier in value_patterns:
                match = re.search(pattern, text)
                if match:
                    value_str = match.group(1).replace(',', '')
                    try:
                        value = float(value_str) * multiplier
                        return np.log10(max(value, 1))
                    except ValueError:
                        continue
        except Exception:
            pass
        return 0.0

    def get_client_score(self, client: str) -> float:
        """🏢 Score client preference"""
        try:
            client_lower = str(client).lower()
            high_pref = ['dot', 'txdot', 'airport', 'federal', 'state', 'transportation']
            medium_pref = ['city', 'county', 'municipal', 'government']
            
            if any(term in client_lower for term in high_pref):
                return 1.0
            elif any(term in client_lower for term in medium_pref):
                return 0.7
            else:
                return 0.3
        except:
            return 0.3

    def get_location_score(self, location: str) -> float:
        """📍 Score location preference"""
        try:
            location_lower = str(location).lower()
            texas_terms = ['texas', 'tx', 'austin', 'houston', 'dallas', 'san antonio']
            southwest_terms = ['arizona', 'new mexico', 'oklahoma', 'arkansas', 'louisiana']
            
            if any(term in location_lower for term in texas_terms):
                return 1.0
            elif any(term in location_lower for term in southwest_terms):
                return 0.8
            else:
                return 0.5
        except:
            return 0.5

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """📊 Prepare data for training"""
        print("🔧 Preparing features...")
        
        try:
            # Extract text features
            text_features = self.extract_text_features(df)
            print(f"   Text features shape: {text_features.shape}")
            
            # Extract numerical features
            numerical_features = self.extract_features(df)
            print(f"   Numerical features shape: {numerical_features.shape}")
            
            # Scale numerical features
            if numerical_features.size == 0:
                raise ValueError("No numerical features extracted")
                
            if self.is_trained:
                try:
                    numerical_features_scaled = self.feature_scaler.transform(numerical_features)
                except:
                    print("⚠️ Error transforming features. Using fallback scaling.")
                    numerical_features_scaled = (numerical_features - np.mean(numerical_features, axis=0)) / (np.std(numerical_features, axis=0) + 1e-6)
            else:
                try:
                    numerical_features_scaled = self.feature_scaler.fit_transform(numerical_features)
                except:
                    print("⚠️ Error fitting scaler. Using fallback scaling.")
                    numerical_features_scaled = (numerical_features - np.mean(numerical_features, axis=0)) / (np.std(numerical_features, axis=0) + 1e-6)
            
            # Combine features
            X = np.hstack([text_features, numerical_features_scaled])
            y = df['label'].values
            
            print(f"   Final feature shape: {X.shape}")
            print(f"   Labels shape: {y.shape}")
            
            return X, y
        except Exception as e:
            print(f"❌ Error preparing training data: {e}")
            raise

    def train_model(self, training_data: pd.DataFrame, validation_split: float = 0.2):
        """🎓 Train the classification model"""
        print("🚀 Starting model training...")
        
        try:
            # Validate data
            training_data = self.validate_training_data(training_data)
            
            # Prepare data
            X, y = self.prepare_training_data(training_data)
            
            # Check data
            unique_labels = np.unique(y)
            if len(unique_labels) < 2:
                raise ValueError("Need at least 2 different classes for training")
            
            # Count samples per class
            samples_per_class = {}
            for label in unique_labels:
                samples_per_class[int(label)] = np.sum(y == label)
                
            min_samples_per_class = min(samples_per_class.values())
            print(f"📊 Smallest class has {min_samples_per_class} samples")
            
            # Adjust validation split for small datasets
            if len(training_data) < 20:
                # For very small datasets, use a smaller validation set
                adjusted_split = 0.1
                print(f"⚠️ Small dataset detected: Reducing validation split from {validation_split} to {adjusted_split}")
                validation_split = adjusted_split
            
            # Split data with appropriate handling for small datasets
            try:
                if len(training_data) >= 10 and min_samples_per_class >= 2:
                    # Normal stratified split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=validation_split, random_state=42, stratify=y
                    )
                    print("📊 Using stratified train/test split")
                else:
                    # Simple random split for very small datasets
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=validation_split, random_state=42
                    )
                    print("⚠️ Using simple random train/test split (dataset too small for stratification)")
            except ValueError as e:
                print(f"⚠️ Error in train/test split: {e}")
                print("⚠️ Falling back to simple random split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=validation_split, random_state=42
                )
            
            # Adjust model complexity based on dataset size
            n_estimators = min(100, max(10, len(X_train) // 2))
            max_depth = min(10, max(3, int(np.log2(len(X_train) + 1))))
            
            print(f"🔧 Adjusting model parameters: n_estimators={n_estimators}, max_depth={max_depth}")
            
            # Train Random Forest model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=2,
                random_state=42,
                class_weight='balanced'
            )
            
            print("🔧 Training Random Forest...")
            
            # Cross-validation with error handling
            min_cv_folds = 2
            max_cv_folds = 5
            # Determine appropriate CV folds based on data size and class balance
            samples_per_fold = min(samples_per_class.values())
            cv_folds = min(max_cv_folds, max(min_cv_folds, samples_per_fold))
            
            if cv_folds >= min_cv_folds and len(X_train) >= cv_folds * 2:
                try:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                    cv_score = cv_scores.mean()
                    print(f"   CV Score: {cv_score:.3f} (+/- {cv_scores.std() * 2:.3f})")
                except Exception as e:
                    print(f"⚠️ Cross-validation failed: {e}")
                    print("⚠️ Skipping cross-validation")
                    cv_score = 0.0
            else:
                cv_score = 0.0
                print(f"⚠️ Skipping CV due to small dataset (need at least {min_cv_folds} samples per class per fold)")
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Test evaluation
            if len(X_test) > 0:
                test_score = model.score(X_test, y_test)
                print(f"📊 Test set accuracy: {test_score:.3f}")
                
                # Classification report - handle edge cases
                try:
                    y_pred = model.predict(X_test)
                    print("\n📋 Classification Report:")
                    target_names = [self.business_lines[int(i)] for i in unique_labels]
                    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
                except Exception as e:
                    print(f"⚠️ Error generating classification report: {e}")
                    
            else:
                test_score = cv_score
                print("⚠️ No test data available")
            
            self.model = model
            self.is_trained = True
            
            # Save model
            save_result = self.save_model()
            if save_result:
                print("💾 Model saved successfully")
            else:
                print("⚠️ Failed to save model")
            
            return {
                'timestamp': datetime.now(),
                'cv_score': cv_score,
                'test_score': test_score,
                'model_type': 'random_forest',
                'training_samples': len(training_data),
                'feature_count': X.shape[1]
            }
        
        except Exception as e:
            print(f"❌ Training error: {e}")
            raise

    def train_multiple_models(self, training_data: pd.DataFrame, validation_split: float = 0.2):
        """🔍 Train and compare multiple classification models"""
        print("🚀 Starting model comparison...")
        
        # Validate and prepare data
        training_data = self.validate_training_data(training_data)
        X, y = self.prepare_training_data(training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42, 
            stratify=y if len(np.unique(y)) <= len(y)//2 else None
        )
        
        # Define models to evaluate
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight='balanced', random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0, max_iter=1000, class_weight='balanced', random_state=42, multi_class='multinomial'
            ),
            'svm': SVC(
                C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=42
            ),
            'decision_tree': DecisionTreeClassifier(
                max_depth=10, class_weight='balanced', random_state=42
            )
        }
        
        # Train and evaluate each model
        results = {}
        best_score = 0
        best_model_name = None
        
        print("\n📊 Model Comparison Results:")
        print("-" * 50)
        print(f"{'Model':<20} {'CV Score':<10} {'Test Score':<10} {'Training Time':<15}")
        print("-" * 50)
        
        for name, model in models.items():
            try:
                # Time the training
                start_time = datetime.now()
                
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                cv_score = cv_scores.mean()
                
                # Test accuracy
                test_score = model.score(X_test, y_test)
                
                # Calculate training time
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Store results
                results[name] = {
                    'model': model,
                    'cv_score': cv_score,
                    'test_score': test_score,
                    'training_time': training_time
                }
                
                # Print results
                print(f"{name:<20} {cv_score:.3f}      {test_score:.3f}      {training_time:.2f}s")
                
                # Track best model
                if test_score > best_score:
                    best_score = test_score
                    best_model_name = name
                    
            except Exception as e:
                print(f"{name:<20} Failed: {str(e)}")
        
        print("-" * 50)
        print(f"🏆 Best model: {best_model_name} with test score: {best_score:.3f}")
        
        # Detailed analysis of best model
        if best_model_name:
            best_model = results[best_model_name]['model']
            y_pred = best_model.predict(X_test)
            
            print("\n📋 Classification Report for Best Model:")
            unique_labels = np.unique(y)
            target_names = [self.business_lines[int(i)] for i in unique_labels]
            print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
            
            # Set as current model
            self.model = best_model
            self.is_trained = True
            
            # Save the model
            save_result = self.save_model()
            if save_result:
                print(f"💾 Best model ({best_model_name}) saved successfully")
            
            # Return best model results
            return {
                'model_type': best_model_name,
                'cv_score': results[best_model_name]['cv_score'],
                'test_score': results[best_model_name]['test_score'],
                'training_samples': len(training_data),
                'feature_count': X.shape[1],
                'all_results': results
            }
        else:
            raise ValueError("No models were successfully trained")

    def predict(self, project_data: Dict) -> Dict:
        """🔮 Predict project relevance"""
        try:
            if not self.is_trained:
                try:
                    if self.load_model():
                        print("📁 Loaded existing model")
                    else:
                        raise ValueError("No trained model available. Please train first.")
                except Exception as e:
                    raise ValueError(f"Model must be trained before making predictions: {e}")
            
            # Validate input
            if not isinstance(project_data, dict):
                raise TypeError("Project data must be a dictionary")
            
            # Ensure project_name exists
            if 'project_name' not in project_data or not project_data['project_name']:
                raise ValueError("Project data must include a non-empty 'project_name'")
            
            # Convert to DataFrame and ensure all expected fields exist
            df = pd.DataFrame([project_data])
            
            # Make sure all expected fields exist (fill with empty strings if not)
            required_fields = ['project_name', 'description', 'client', 'location', 'project_value']
            for field in required_fields:
                if field not in df.columns:
                    df[field] = ''
                else:
                    df[field] = df[field].fillna('')
            
            # Extract features
            try:
                text_features = self.extract_text_features(df)
            except Exception as e:
                print(f"⚠️ Error extracting text features: {e}")
                # Create empty array with correct dimensions
                if hasattr(self.vectorizer, 'get_feature_names_out'):
                    text_features = np.zeros((1, len(self.vectorizer.get_feature_names_out())))
                else:
                    text_features = np.zeros((1, 500))  # Fallback dimension
            
            try:
                numerical_features = self.extract_features(df)
                numerical_features_scaled = self.feature_scaler.transform(numerical_features)
            except Exception as e:
                print(f"⚠️ Error extracting numerical features: {e}")
                numerical_features_scaled = np.zeros((1, len(self.keyword_categories) + 6))
            
            # Combine features
            X = np.hstack([text_features, numerical_features_scaled])
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            # Get confidence scores for each business line
            business_line_scores = {}
            for i, prob in enumerate(probability):
                business_line = self.business_lines.get(i, f"unknown_{i}")
                business_line_scores[business_line] = float(prob)
            
            return {
                'predicted_class': self.business_lines[prediction],
                'confidence': float(max(probability)),
                'business_line_scores': business_line_scores,
                'recommendation': self.get_recommendation(prediction, max(probability))
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'predicted_class': 'not_relevant',
                'confidence': 0.0,
                'business_line_scores': {},
                'recommendation': 'ERROR - Could not analyze project',
                'error': str(e)
            }

    def get_recommendation(self, prediction: int, confidence: float) -> str:
        """💡 Generate recommendation based on prediction"""
        if prediction == 0:
            return "NOT RECOMMENDED - Outside core competencies"
        elif confidence >= 0.8:
            return "HIGH PRIORITY - Pursue aggressively"
        elif confidence >= 0.6:
            return "MEDIUM PRIORITY - Consider pursuing"
        else:
            return "LOW PRIORITY - Monitor opportunity"

    def save_model(self):
        """💾 Save trained model and components"""
        if not self.is_trained or self.model is None:
            print("⚠️ Cannot save - model is not trained")
            return False
            
        try:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'feature_scaler': self.feature_scaler,
                'business_lines': self.business_lines,
                'is_trained': self.is_trained,
                'keyword_categories': self.keyword_categories,
                'version': '1.0'
            }
            
            try:
                with open(self.model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                
                print(f"💾 Model saved to {self.model_path}")
                return True
            except PermissionError:
                alt_path = f"transtec_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                print(f"⚠️ Permission denied. Trying alternative path: {alt_path}")
                with open(alt_path, 'wb') as f:
                    pickle.dump(model_data, f)
                print(f"💾 Model saved to {alt_path}")
                self.model_path = alt_path
                return True
                
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

    def load_model(self):
        """📁 Load trained model and components"""
        try:
            if not os.path.exists(self.model_path):
                print(f"⚠️ Model file not found: {self.model_path}")
                return False
                
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Validate model data
            required_keys = ['model', 'vectorizer', 'feature_scaler', 'business_lines']
            missing_keys = [key for key in required_keys if key not in model_data]
            
            if missing_keys:
                print(f"⚠️ Invalid model file: missing {', '.join(missing_keys)}")
                return False
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.feature_scaler = model_data['feature_scaler']
            self.business_lines = model_data['business_lines']
            self.is_trained = model_data.get('is_trained', True)
            
            if 'keyword_categories' in model_data:
                self.keyword_categories = model_data['keyword_categories']
            
            print(f"📁 Model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

# Sample training data for testing
def create_sample_training_data() -> pd.DataFrame:
    """📊 Create sample training data"""
    sample_data = [
        # Research (Label 1)
        {'project_name': 'HIPERPAV Technology Research', 'description': 'Advanced research study on high performance pavement technology', 'client': 'Federal Highway Administration', 'location': 'Austin, Texas', 'project_value': '$750,000', 'label': 1},
        {'project_name': 'Pavement Performance Study', 'description': 'Long-term research investigation on pavement durability', 'client': 'TxDOT Research Division', 'location': 'Texas', 'project_value': '$500,000', 'label': 1},
        {'project_name': 'Innovative Materials Research', 'description': 'Experimental study on new pavement materials', 'client': 'University Research Center', 'location': 'College Station, Texas', 'project_value': '$400,000', 'label': 1},
        {'project_name': 'Advanced Testing Protocols', 'description': 'Research and development of new testing protocols', 'client': 'Research Institute', 'location': 'Austin, Texas', 'project_value': '$350,000', 'label': 1},
        
        # Engineering (Label 2)
        {'project_name': 'I-35 Pavement Design', 'description': 'Complete engineering design of concrete pavement', 'client': 'TxDOT', 'location': 'Austin, Texas', 'project_value': '$15 million', 'label': 2},
        {'project_name': 'Airport Runway Engineering', 'description': 'Structural engineering design for new runway pavement', 'client': 'DFW Airport', 'location': 'Dallas, Texas', 'project_value': '$25 million', 'label': 2},
        {'project_name': 'Highway Rehabilitation Design', 'description': 'Engineering design for highway pavement rehabilitation', 'client': 'State DOT', 'location': 'Houston, Texas', 'project_value': '$18 million', 'label': 2},
        {'project_name': 'Concrete Pavement Engineering', 'description': 'Detailed engineering design and specifications for concrete pavement', 'client': 'Texas Transportation Commission', 'location': 'San Antonio, Texas', 'project_value': '$12 million', 'label': 2},
        
        # Testing (Label 3)
        {'project_name': 'FWD Testing Services', 'description': 'Non-destructive testing using falling weight deflectometer', 'client': 'City of Houston', 'location': 'Houston, Texas', 'project_value': '$300,000', 'label': 3},
        {'project_name': 'Pavement Condition Assessment', 'description': 'Comprehensive testing and evaluation of pavement condition', 'client': 'TxDOT', 'location': 'San Antonio, Texas', 'project_value': '$400,000', 'label': 3},
        {'project_name': 'Airport Testing Program', 'description': 'Structural testing and evaluation of airport pavements', 'client': 'Austin Airport Authority', 'location': 'Austin, Texas', 'project_value': '$275,000', 'label': 3},
        {'project_name': 'GPR Survey Services', 'description': 'Ground penetrating radar testing for pavement evaluation', 'client': 'City of Dallas', 'location': 'Dallas, Texas', 'project_value': '$150,000', 'label': 3},
        
        # Tracks (Label 4)
        {'project_name': 'Texas Motor Speedway Surface', 'description': 'Specialized track surface design for motorsports racing', 'client': 'Texas Motor Speedway', 'location': 'Fort Worth, Texas', 'project_value': '$5 million', 'label': 4},
        {'project_name': 'Racing Circuit Development', 'description': 'Design and engineering of new racetrack surface', 'client': 'Private Racing Facility', 'location': 'Austin, Texas', 'project_value': '$8 million', 'label': 4},
        {'project_name': 'Motorsports Track Rehabilitation', 'description': 'Complete rehabilitation of existing racing track surface', 'client': 'NASCAR Facility', 'location': 'Texas', 'project_value': '$6 million', 'label': 4},
        {'project_name': 'High Performance Track Design', 'description': 'Engineering design for high-speed test track', 'client': 'Automotive Testing Facility', 'location': 'Houston, Texas', 'project_value': '$4 million', 'label': 4},
        
        # Not Relevant (Label 0)
        {'project_name': 'Office Building Construction', 'description': 'General building construction and HVAC installation', 'client': 'Private Developer', 'location': 'Houston, Texas', 'project_value': '$3 million', 'label': 0},
        {'project_name': 'Water Treatment Plant', 'description': 'Municipal water treatment facility design', 'client': 'City of Dallas', 'location': 'Dallas, Texas', 'project_value': '$10 million', 'label': 0},
        {'project_name': 'Shopping Mall Development', 'description': 'Commercial retail development project', 'client': 'Retail Developer', 'location': 'Austin, Texas', 'project_value': '$15 million', 'label': 0},
        {'project_name': 'Residential Complex', 'description': 'Multi-family residential building construction', 'client': 'Housing Developer', 'location': 'San Antonio, Texas', 'project_value': '$8 million', 'label': 0}
    ]
    
    return pd.DataFrame(sample_data)

def test_with_excel_data():
    """🧪 Test classifier with Excel data"""
    print("🧪 Testing Transtec ML Classifier")
    print("=" * 50)
    
    # Initialize classifier with error handling
    try:
        classifier = TranstecMLClassifier()
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return
    
    # Excel file path
    excel_path = r"C:\Users\aanand\OneDrive - Terracon Consultants Inc\Desktop\PythonProjects\Agents\project training data.xlsx"
    
    # Try to load Excel data
    training_data = None
    if os.path.exists(excel_path):
        print(f"📁 Found Excel file at: {excel_path}")
        try:
            training_data = classifier.load_training_data_from_excel(excel_path)
            print("✅ Successfully loaded Excel data!")
        except PermissionError:
            print("❌ Permission denied - the file might be open in Excel or another program")
            print("📋 Using sample data instead...")
        except Exception as e:
            print(f"❌ Error loading Excel data: {e}")
            print("📋 Using sample data instead...")
    else:
        print(f"❌ Excel file not found: {excel_path}")
        print("📋 Using sample data instead...")
    
    # Use sample data if Excel loading failed
    if training_data is None:
        try:
            training_data = create_sample_training_data()
            print(f"📊 Created {len(training_data)} sample training records")
        except Exception as e:
            print(f"❌ Failed to create sample data: {e}")
            return
    
    # Train the model
    print(f"\n🎓 Training model with {len(training_data)} samples...")
    try:
        results = classifier.train_model(training_data)
        print(f"✅ Training completed successfully!")
        print(f"   Model type: {results['model_type']}")
        print(f"   CV Score: {results['cv_score']:.3f}")
        print(f"   Test Score: {results['test_score']:.3f}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Test predictions
    print(f"\n🔮 Testing predictions...")
    test_projects = [
        {
            'project_name': 'Innovative Pavement Materials Study',
            'description': 'Research and development of advanced pavement materials',
            'client': 'Research Institute',
            'location': 'Austin, Texas',
            'project_value': '$600,000',
            'expected': 'Research'
        },
        {
            'project_name': 'Highway Rehabilitation Design',
            'description': 'Complete engineering design for highway pavement rehabilitation',
            'client': 'State DOT',
            'location': 'Texas',
            'project_value': '$12 million',
            'expected': 'Engineering'
        },
        {
            'project_name': 'Airport Runway Testing',
            'description': 'Comprehensive testing using FWD and GPR for runway evaluation',
            'client': 'Airport Authority',
            'location': 'Dallas, Texas',
            'project_value': '$250,000',
            'expected': 'Testing'
        },
        {
            'project_name': 'Motorsports Track Surface',
            'description': 'Specialized racing track surface design for high-speed motorsports',
            'client': 'Racing Facility',
            'location': 'Texas',
            'project_value': '$4 million',
            'expected': 'Tracks'
        }
    ]
    
    for i, project in enumerate(test_projects, 1):
        print(f"\n📋 Test Project {i}: {project['project_name']}")
        print(f"   Expected: {project['expected']}")
        try:
            prediction = classifier.predict(project)
            print(f"   Predicted: {prediction['predicted_class']}")
            print(f"   Confidence: {prediction['confidence']:.1%}")
            print(f"   Recommendation: {prediction['recommendation']}")
            
            # Show top business line scores
            print(f"   Business Line Scores:")
            sorted_scores = sorted(prediction['business_line_scores'].items(), 
                                 key=lambda x: x[1], reverse=True)
            for line, score in sorted_scores[:3]:
                print(f"     {line}: {score:.1%}")
                
            # Check if prediction matches expected
            if prediction['predicted_class'] == project['expected']:
                print(f"   ✅ CORRECT PREDICTION")
            else:
                print(f"   ⚠️ INCORRECT PREDICTION")
                
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")

    # Overall summary
    print("\n📊 Summary of Model Performance")
    print(f"   Model type: {results['model_type']}")
    print(f"   Training samples: {results['training_samples']}")
    print(f"   Feature count: {results['feature_count']}")
    print(f"   CV Score: {results['cv_score']:.3f}")
    print(f"   Test Score: {results['test_score']:.3f}")
    
    print("\n💡 Tips for improvement:")
    print("   1. Add more training samples for each business line")
    print("   2. Improve project descriptions in your training data")
    print("   3. Run the model on your actual project data")
    print("   4. Provide feedback on incorrect predictions")
    
    # Provide instructions for using the trained model
    print("\n🚀 To use this trained model in your applications:")
    print("   1. Import the TranstecMLClassifier class")
    print("   2. Initialize with: classifier = TranstecMLClassifier()")
    print("   3. Make predictions with: result = classifier.predict(project_data)")
    print("   4. Access results with: result['predicted_class'], result['confidence']")

def test_with_multiple_models():
    """🧪 Test classifier with multiple ML models"""
    print("🧪 Testing Transtec ML Classifier with Multiple Models")
    print("=" * 60)
    
    # Initialize classifier
    classifier = TranstecMLClassifier()
    
    # Excel file path
    excel_path = r"C:\Users\aanand\OneDrive - Terracon Consultants Inc\Desktop\PythonProjects\Agents\project training data.xlsx"
    
    # Load training data
    if os.path.exists(excel_path):
        print(f"📁 Found Excel file at: {excel_path}")
        try:
            training_data = classifier.load_training_data_from_excel(excel_path)
            print("✅ Successfully loaded Excel data!")
        except Exception as e:
            print(f"❌ Error loading Excel data: {e}")
            training_data = create_sample_training_data()
    else:
        print(f"❌ Excel file not found: {excel_path}")
        training_data = create_sample_training_data()
    
    # Train multiple models
    print(f"\n🎓 Training multiple models with {len(training_data)} samples...")
    try:
        results = classifier.train_multiple_models(training_data)
        print(f"✅ Model comparison completed!")
        print(f"   Best model: {results['model_type']}")
        print(f"   CV Score: {results['cv_score']:.3f}")
        print(f"   Test Score: {results['test_score']:.3f}")
        
        # Test the best model with sample projects
        print(f"\n🔮 Testing predictions with best model ({results['model_type']})...")
        test_projects = [
            {
                'project_name': 'Innovative Pavement Materials Study',
                'description': 'Research and development of advanced pavement materials',
                'client': 'Research Institute',
                'location': 'Austin, Texas',
                'project_value': '$600,000',
                'expected': 'Research'
            },
            {
                'project_name': 'Highway Rehabilitation Design',
                'description': 'Complete engineering design for highway pavement rehabilitation',
                'client': 'State DOT',
                'location': 'Texas',
                'project_value': '$12 million',
                'expected': 'Engineering'
            },
            {
                'project_name': 'Airport Runway Testing',
                'description': 'Comprehensive testing using FWD and GPR for runway evaluation',
                'client': 'Airport Authority',
                'location': 'Dallas, Texas',
                'project_value': '$250,000',
                'expected': 'Testing'
            },
            {
                'project_name': 'Motorsports Track Surface',
                'description': 'Specialized racing track surface design for high-speed motorsports',
                'client': 'Racing Facility',
                'location': 'Texas',
                'project_value': '$4 million',
                'expected': 'Tracks'
            }
        ]
        
        for i, project in enumerate(test_projects, 1):
            print(f"\n📋 Test Project {i}: {project['project_name']}")
            print(f"   Expected: {project['expected']}")
            try:
                prediction = classifier.predict(project)
                print(f"   Predicted: {prediction['predicted_class']}")
                print(f"   Confidence: {prediction['confidence']:.1%}")
                print(f"   Recommendation: {prediction['recommendation']}")
                
                # Show top business line scores
                print(f"   Business Line Scores:")
                sorted_scores = sorted(prediction['business_line_scores'].items(), 
                                     key=lambda x: x[1], reverse=True)
                for line, score in sorted_scores[:3]:
                    print(f"     {line}: {score:.1%}")
                    
                # Check if prediction matches expected
                if prediction['predicted_class'] == project['expected']:
                    print(f"   ✅ CORRECT PREDICTION")
                else:
                    print(f"   ⚠️ INCORRECT PREDICTION")
                    
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")

        # Overall summary
        print("\n📊 Summary of Model Performance")
        print(f"   Model type: {results['model_type']}")
        print(f"   Training samples: {results['training_samples']}")
        print(f"   Feature count: {results['feature_count']}")
        print(f"   CV Score: {results['cv_score']:.3f}")
        print(f"   Test Score: {results['test_score']:.3f}")
        
        print("\n💡 Tips for improvement:")
        print("   1. Add more training samples for each business line")
        print("   2. Improve project descriptions in your training data")
        print("   3. Run the model on your actual project data")
        print("   4. Provide feedback on incorrect predictions")
        
        # Provide instructions for using the trained model
        print("\n🚀 To use this trained model in your applications:")
        print("   1. Import the TranstecMLClassifier class")
        print("   2. Initialize with: classifier = TranstecMLClassifier()")
        print("   3. Make predictions with: result = classifier.predict(project_data)")
        print("   4. Access results with: result['predicted_class'], result['confidence']")
        
    except Exception as e:
        print(f"❌ Model comparison failed: {e}")

if __name__ == "__main__":
    # test_with_excel_data()
    test_with_multiple_models()