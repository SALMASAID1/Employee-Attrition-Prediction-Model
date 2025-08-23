import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Model Management Functions ---
@st.cache_resource
def load_models():
    """Load models from disk with caching"""
    models_dir = "saved_models"
    
    try:
        rf_model = joblib.load(f"{models_dir}/random_forest_model.pkl")
        logreg_model = joblib.load(f"{models_dir}/logistic_regression_model.pkl")
        kmeans_model = joblib.load(f"{models_dir}/kmeans_model.pkl")
        scaler = joblib.load(f"{models_dir}/scaler_model.pkl")
        training_info = joblib.load(f"{models_dir}/training_info.pkl")
        
        return rf_model, logreg_model, kmeans_model, scaler, training_info, True
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        return None, None, None, None, None, False
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None, None, False

def train_and_save_models():
    """Train new models and save them"""
    models_dir = "saved_models"
    
    try:
        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Load data for training
        df = pd.read_csv("data/encoded_employee_attrition.csv")
        X = df.drop('Attrition_Yes', axis=1)
        y = df['Attrition_Yes']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train Random Forest
        status_text.text("Training Random Forest...")
        progress_bar.progress(25)
        rf_new = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_new.fit(X_train, y_train)
        
        # Train Logistic Regression
        status_text.text("Training Logistic Regression...")
        progress_bar.progress(50)
        logreg_new = LogisticRegression(max_iter=1000, random_state=42)
        logreg_new.fit(X_train, y_train)
        
        # Train K-Means and Scaler
        status_text.text("Training K-Means and Scaler...")
        progress_bar.progress(75)
        scaler_new = StandardScaler()
        X_scaled = scaler_new.fit_transform(X)
        kmeans_new = KMeans(n_clusters=2, random_state=42)
        kmeans_new.fit(X_scaled)
        
        # Save models
        status_text.text("Saving models...")
        progress_bar.progress(90)
        joblib.dump(rf_new, f"{models_dir}/random_forest_model.pkl")
        joblib.dump(logreg_new, f"{models_dir}/logistic_regression_model.pkl")
        joblib.dump(kmeans_new, f"{models_dir}/kmeans_model.pkl")
        joblib.dump(scaler_new, f"{models_dir}/scaler_model.pkl")
        
        training_info = {
            'feature_names': X.columns.tolist(),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        joblib.dump(training_info, f"{models_dir}/training_info.pkl")
        
        progress_bar.progress(100)
        status_text.text("✅ Models trained and saved successfully!")
        
        # Clear cache to reload models
        load_models.clear()
        
        return True
        
    except FileNotFoundError:
        st.error("❌ Training data not found. Please ensure 'data/encoded_employee_attrition.csv' exists.")
        return False
    except Exception as e:
        st.error(f"❌ Error training models: {e}")
        return False

# --- Load Models ---
rf_model, logreg_model, kmeans_model, scaler, training_info, models_loaded = load_models()

# --- Page Setup ---
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="💼", layout="wide")

# --- Model Status Check ---
if not models_loaded:
    st.error("🚨 **Models Not Loaded!**")
    st.warning("The prediction models could not be loaded. This could be because:")
    st.write("• Model files are missing from the 'saved_models' directory")
    st.write("• Model files are corrupted")
    st.write("• This is the first time running the app")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Try Loading Models Again"):
            load_models.clear()
            st.rerun()
    
    with col2:
        if st.button("🏗️ Train New Models"):
            if train_and_save_models():
                st.rerun()
    
    st.stop()  # Stop execution if models aren't loaded

# Extract feature names from training info
feature_names = training_info['feature_names']

# --- Model Information Sidebar ---
with st.sidebar:
    st.header(" Model Information")
    st.success("✅ All models loaded successfully!")
    
    if st.button("🔄 Reload Models"):
        load_models.clear()
        st.rerun()
    
    if st.button("🏗️ Retrain Models"):
        if train_and_save_models():
            st.rerun()
    
    st.write(f"**Features:** {len(feature_names)}")
    st.write(f"**Training samples:** {training_info['train_size']}")
    st.write(f"**Test samples:** {training_info['test_size']}")
    
    with st.expander("View All Features"):
        for i, feature in enumerate(feature_names, 1):
            st.write(f"{i}. {feature}")
    
    # Model performance metrics (if available)
    with st.expander("Model Performance"):
        st.info("Model metrics from last training session")
        st.write(" **Random Forest**")
        st.write("• High accuracy for complex patterns")
        st.write("• Feature importance analysis")
        st.write("")
        st.write(" **Logistic Regression**")
        st.write("• Fast and interpretable")
        st.write("• Good baseline performance")
        st.write("")
        st.write(" **K-Means Clustering**")
        st.write("• Employee segmentation")
        st.write("• Risk group identification")

# Custom CSS for better style
st.markdown("""
    <style>
        /* Full-page background */
        .stApp {
            background: url('https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=1950&q=80') no-repeat center center fixed;
            background-size: cover;
        }
        /* Main content container with dark overlay */
        .block-container {
            background-color: rgba(34, 34, 34, 0.85);
            padding: 20px;
            border-radius: 15px;
            color: #ffffff;
        }
        /* Title style with text shadow for extra pop */
        .stTitle {
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
        }
        /* Prediction card styling */
        .prediction-card {
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            color: #ffffff;
            font-size: 18px;
            margin-bottom: 15px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        }
        .leave { background-color: #e74c3c; }
        .stay { background-color: #27ae60; }
        .risk { background-color: #2980b9; }
        .very-high-risk { background-color: #c0392b; }
        .high-risk { background-color: #e67e22; }
        .medium-risk { background-color: #f1c40f; color: #000000; }
        .low-risk { background-color: #27ae60; }
        /* Style form headers and labels */
        .css-1d391kg { color: #ffffff; }
        .stSelectbox label { color: #ffffff !important; }
        .stNumberInput label { color: #ffffff !important; }
        .stSlider label { color: #ffffff !important; }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("💼 Employee Attrition Prediction System")
st.write("**Advanced AI-powered prediction using Random Forest & Logistic Regression models**")
st.write("Get instant attrition risk assessment with K-Means clustering insights.")

# --- Layout: 2 columns ---
col1, col2 = st.columns([1.2, 2])

# --- Input Form (Left Side) ---
with col1:
    st.subheader("👤Employee Information")
    
    # Personal Information
    st.write("**Personal Details**")
    age = st.number_input("Age", min_value=18, max_value=65, value=30, help="Employee's current age", key="age")
    distance = st.slider("Distance from Home (km)", 1, 50, 10, help="Commute distance to workplace", key="distance")
    
    # Job Information
    st.write("**Job Details**")
    income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100, key="income")
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3, key="years_company")
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=2, help="1=Entry, 5=Executive", key="job_level")
    
    # Satisfaction Metrics
    st.write("**Satisfaction & Work-Life**")
    job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2, 
                                  help="1=Low, 2=Medium, 3=High, 4=Very High", key="job_sat")
    work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4], index=2,
                                   help="1=Bad, 2=Good, 3=Better, 4=Best", key="work_life")
    overtime = st.selectbox("Works Overtime", ["Yes", "No"], index=1, key="overtime")
    
    # Advanced Options - Now with proper dynamic max values and validation
    with st.expander("🔧 Advanced Options"):
        # Total Working Years - should be at least as much as years at company
        min_total_years = max(years_at_company, age - 22 if age >= 22 else 1)
        total_working_years = st.number_input("Total Working Years", 
                                            min_value=min_total_years, 
                                            max_value=50, 
                                            value=max(min_total_years, age-22 if age >= 22 else 1),
                                            help=f"Must be at least {min_total_years} (Years at Company or Age-22)",
                                            key="total_years")
        
        # Years in Current Role with proper validation
        max_current_role = min(years_at_company, total_working_years)  # Cannot exceed either value
        default_current_role = min(2, max_current_role) if max_current_role > 0 else 0
        
        years_current_role = st.number_input("Years in Current Role", 
                                           min_value=0, 
                                           max_value=max_current_role, 
                                           value=default_current_role,
                                           help=f"Cannot exceed Years at Company ({years_at_company}) or Total Working Years ({total_working_years})",
                                           key="current_role")
        
        # Validation warnings for all time relationships
        validation_warnings = []
        if years_current_role > years_at_company:
            validation_warnings.append(f"Years in Current Role ({years_current_role}) cannot be greater than Years at Company ({years_at_company})")
        if years_current_role > total_working_years:
            validation_warnings.append(f"Years in Current Role ({years_current_role}) cannot be greater than Total Working Years ({total_working_years})")
        if total_working_years < years_at_company:
            validation_warnings.append(f"Total Working Years ({total_working_years}) should be at least equal to Years at Company ({years_at_company})")
        
        if validation_warnings:
            for warning in validation_warnings:
                st.warning(f"⚠️ {warning}")
        
        years_since_promotion = st.number_input("Years Since Last Promotion", 
                                              min_value=0, 
                                              max_value=min(years_at_company, years_current_role) if years_current_role > 0 else years_at_company, 
                                              value=min(1, years_at_company),
                                              help=f"Cannot exceed Years in Current Role ({years_current_role})",
                                              key="promotion")
        environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2, key="env_sat")
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2, key="job_inv")
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=0, key="stock")

    # Submit button outside the form for better interactivity
    submitted = st.button("🔮 Predict Attrition Risk", use_container_width=True, type="primary")

# --- Prediction (Right Side) ---
with col2:
    if submitted:
        # Comprehensive validation check for all time-related variables
        validation_errors = []
        
        # Years in Current Role validations
        if years_current_role > years_at_company:
            validation_errors.append(f"Years in Current Role ({years_current_role}) cannot be greater than Years at Company ({years_at_company})")
        if years_current_role > total_working_years:
            validation_errors.append(f"Years in Current Role ({years_current_role}) cannot be greater than Total Working Years ({total_working_years})")
        
        # Total Working Years validations
        if total_working_years < years_at_company:
            validation_errors.append(f"Total Working Years ({total_working_years}) should be at least equal to Years at Company ({years_at_company})")
        
        # Years Since Last Promotion validations
        if years_since_promotion > years_current_role and years_current_role > 0:
            validation_errors.append(f"Years Since Last Promotion ({years_since_promotion}) cannot be greater than Years in Current Role ({years_current_role})")
        if years_since_promotion > years_at_company:
            validation_errors.append(f"Years Since Last Promotion ({years_since_promotion}) cannot be greater than Years at Company ({years_at_company})")
        
        # Age and experience validations
        if total_working_years > (age - 16):  # Assuming minimum working age of 16
            validation_errors.append(f"Total Working Years ({total_working_years}) seems too high for age {age}. Maximum reasonable: {age - 16}")
        
        if validation_errors:
            st.error("❌ **Validation Errors - Please Fix These Issues:**")
            for i, error in enumerate(validation_errors, 1):
                st.write(f"{i}. {error}")
            st.info("💡 **Tips:**")
            st.write("• Total Working Years ≥ Years at Company ≥ Years in Current Role")
            st.write("• Years Since Last Promotion ≤ Years in Current Role")
            st.write("• All values should be realistic for the employee's age")
        else:
            # Create input dictionary with all features
            input_dict = {f: 0 for f in feature_names}
            
            # Update with user inputs
            input_dict.update({
                "Age": age,
                "MonthlyIncome": income,
                "YearsAtCompany": years_at_company,
                "JobSatisfaction": job_satisfaction,
                "WorkLifeBalance": work_life_balance,
                "OverTime_Yes": 1 if overtime == "Yes" else 0,
                "JobLevel": job_level,
                "DistanceFromHome": distance,
                "TotalWorkingYears": total_working_years,
                "YearsInCurrentRole": years_current_role,
                "YearsSinceLastPromotion": years_since_promotion,
                "EnvironmentSatisfaction": environment_satisfaction,
                "JobInvolvement": job_involvement,
                "StockOptionLevel": stock_option_level
                })
            
            # Fill any missing features with median values (in a real app, you'd use training data medians)
            default_values = {
                "BusinessTravel_Travel_Frequently": 0,
                "BusinessTravel_Travel_Rarely": 1,
                "Department_Research & Development": 1,
                "Department_Sales": 0,
                "Education": 3,
                "EducationField_Life Sciences": 1,
                "EducationField_Medical": 0,
                "EducationField_Marketing": 0,
                "EducationField_Other": 0,
                "EducationField_Technical Degree": 0,
                "Gender_Male": 1,
                "MaritalStatus_Married": 1,
                "MaritalStatus_Single": 0,
                "NumCompaniesWorked": 2,
                "PercentSalaryHike": 15,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 3,
                "StandardHours": 80,
                "TrainingTimesLastYear": 3,
                "YearsWithCurrManager": 2
            }
            
            # Update with defaults for missing features
            for feature, default_value in default_values.items():
                if feature in input_dict and input_dict[feature] == 0:
                    input_dict[feature] = default_value
            
            employee_df = pd.DataFrame([input_dict])
            
            # Scale for clustering
            X_scaled = scaler.transform(employee_df)
            
            # Make predictions
            with st.spinner("Analyzing employee data..."):
                rf_pred = rf_model.predict(employee_df)[0]
                rf_prob = rf_model.predict_proba(employee_df)[0][1]
                
                lr_pred = logreg_model.predict(employee_df)[0]
                lr_prob = logreg_model.predict_proba(employee_df)[0][1]
                
                cluster = kmeans_model.predict(X_scaled)[0]
                
                # Calculate consensus risk
                avg_prob = (rf_prob + lr_prob) / 2
                
                if avg_prob > 0.7:
                    risk = "VERY HIGH"
                    risk_class = "very-high-risk"
                    risk_emoji = "🚨"
                    risk_message = "Immediate attention and retention strategies needed!"
                elif avg_prob > 0.5:
                    risk = "HIGH"
                    risk_class = "high-risk"
                    risk_emoji = "⚠️"
                    risk_message = "Consider implementing retention strategies soon."
                elif avg_prob > 0.3:
                    risk = "MEDIUM"
                    risk_class = "medium-risk"
                    risk_emoji = "⚡"
                    risk_message = "Monitor employee engagement and satisfaction."
                else:
                    risk = "LOW"
                    risk_class = "low-risk"
                    risk_emoji = "✅"
                    risk_message = "Employee likely to stay. Maintain current conditions."

            # --- Results Display ---
            st.subheader(" Prediction Results")

            # Individual Model Predictions
            col_rf, col_lr = st.columns(2)
            
            with col_rf:
                st.markdown(
                    f'<div class="prediction-card {"leave" if rf_pred==1 else "stay"}">'
                    f'🌳 <strong>Random Forest</strong><br>'
                    f'Prediction: {"Will Leave" if rf_pred==1 else "Will Stay"}<br>'
                    f'Confidence: {rf_prob:.1%}</div>', 
                    unsafe_allow_html=True)

            with col_lr:
                st.markdown(
                    f'<div class="prediction-card {"leave" if lr_pred==1 else "stay"}">'
                    f'📊 <strong>Logistic Regression</strong><br>'
                    f'Prediction: {"Will Leave" if lr_pred==1 else "Will Stay"}<br>'
                    f'Confidence: {lr_prob:.1%}</div>', 
                    unsafe_allow_html=True)

            # Consensus Prediction
            st.markdown(
                f'<div class="prediction-card {risk_class}">'
                f'{risk_emoji} <strong>CONSENSUS RISK LEVEL</strong><br>'
                f'<h2 style="margin: 10px 0;">{risk}</h2>'
                f'Average Probability: {avg_prob:.1%}<br>'
                f'<em>{risk_message}</em></div>', 
                unsafe_allow_html=True)

            # K-Means Cluster Information
            cluster_info = {
                0: {"name": "Stable Employees", "description": "Lower attrition risk group"},
                1: {"name": "At-Risk Employees", "description": "Higher attrition risk group"}
            }
            
            cluster_name = cluster_info.get(cluster, {"name": f"Cluster {cluster}", "description": "Employee segment"})
            st.info(f" **Employee Segment:** {cluster_name['name']} - {cluster_name['description']}")

            # --- Visualizations ---
            st.subheader(" Analysis Dashboard")
            
            # Probability Comparison Chart
            prob_df = pd.DataFrame({
                "Model": ["Random Forest", "Logistic Regression", "Consensus"],
                "Leave Probability": [rf_prob, lr_prob, avg_prob],
                "Stay Probability": [1-rf_prob, 1-lr_prob, 1-avg_prob]
            })
            
            fig = px.bar(prob_df, x="Model", y="Leave Probability", 
                         color="Model", range_y=[0, 1], text_auto=".1%",
                         title=" Attrition Probability by Model",
                         color_discrete_sequence=["#e74c3c", "#3498db", "#9b59b6"])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Feature Importance Analysis
            st.subheader("🔍 Key Factors Analysis")
            
            # Get top features influencing this prediction
            feature_importance = pd.DataFrame({
                "Feature": feature_names,
                "Importance": rf_model.feature_importances_,
                "Employee_Value": [input_dict[feature] for feature in feature_names]
            }).sort_values('Importance', ascending=False).head(8)
            
            # Create importance chart
            fig_importance = px.bar(feature_importance, x="Importance", y="Feature", 
                                   orientation='h', title="🎯 Top Factors Influencing Prediction",
                                   text="Importance")
            fig_importance.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Feature values table
            st.write("**Employee's Values for Key Factors:**")
            importance_display = feature_importance[['Feature', 'Employee_Value', 'Importance']].copy()
            importance_display['Employee_Value'] = importance_display['Employee_Value'].round(2)
            importance_display['Importance'] = importance_display['Importance'].round(3)
            importance_display.columns = ['Factor', 'Employee Value', 'Model Importance']
            st.dataframe(importance_display, use_container_width=True)

            # Recommendations based on risk level
            st.subheader(" Recommendations")
            
            if avg_prob > 0.7:
                st.error("**Immediate Action Required:**")
                st.write("• Schedule one-on-one meeting with HR/Manager")
                st.write("• Review compensation and benefits package")
                st.write("• Discuss career development opportunities")
                st.write("• Address work-life balance concerns")
                st.write("• Consider retention bonus or promotion")
            elif avg_prob > 0.5:
                st.warning("**Proactive Measures Recommended:**")
                st.write("• Regular check-ins with manager")
                st.write("• Employee satisfaction survey")
                st.write("• Skill development programs")
                st.write("• Team building activities")
                st.write("• Flexible work arrangements")
            elif avg_prob > 0.3:
                st.info("**Monitor and Maintain:**")
                st.write("• Quarterly performance reviews")
                st.write("• Recognition programs")
                st.write("• Career path discussions")
                st.write("• Maintain current engagement levels")
            else:
                st.success("**Continue Current Strategy:**")
                st.write("• Employee is well-engaged")
                st.write("• Maintain current management approach")
                st.write("• Use as mentor for other employees")
                st.write("• Consider for leadership development")

    else:
        # Show some interesting statistics when no prediction is made
        st.subheader(" 👉🏾 ​About This Prediction System")
        
        col1_info, col2_info = st.columns(2)
        
        with col1_info:
            st.markdown("""
            **AI Models Used:**
            - **Random Forest**: Ensemble learning for complex patterns
            - **Logistic Regression**: Statistical baseline model
            - **K-Means Clustering**: Employee segmentation
            """)
            
        with col2_info:
            st.markdown("""
            **Key Features Analyzed:**
            - Job satisfaction & work-life balance
            - Compensation & career progression  
            - Work environment factors
            - Personal & demographic data
            """)
        
        st.info(" **Enter employee information in the form to get instant attrition risk prediction!**")
        
        # Sample prediction showcase
        with st.expander("🎭 See Sample Predictions"):
            sample_data = [
                {"Profile": "Young Professional 👩‍💼", "Age": 25, "Income": 3000, "Years": 1, "Satisfaction": 2, "Risk": "High"},
                {"Profile": "Experienced Employee 👨‍💼", "Age": 40, "Income": 8000, "Years": 8, "Satisfaction": 4, "Risk": "Low"},
                {"Profile": "Mid-Career Specialist 👩‍🔬", "Age": 35, "Income": 6000, "Years": 5, "Satisfaction": 3, "Risk": "Medium"}
            ]
            
            for sample in sample_data:
                st.write(f"**{sample['Profile']}** - Age: {sample['Age']}, Income: ${sample['Income']}, Experience: {sample['Years']} years")
                st.write(f"Satisfaction: {sample['Satisfaction']}/4, Predicted Risk: **{sample['Risk']}**")
                st.write("---")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #ffffff; opacity: 0.7;'>"
    "💼 Employee Attrition Prediction System | Powered by Advanced Machine Learning"
    "</div>", 
    unsafe_allow_html=True
)
