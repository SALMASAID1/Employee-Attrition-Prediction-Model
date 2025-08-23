from xml.parsers.expat import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import silhouette_score, adjusted_rand_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add joblib and os for model saving
import joblib
import os

# Load your data
df = pd.read_csv(r"C:\Users\pc\Desktop\employee-attrition-prediction\data\encoded_employee_attrition.csv")
X = df.drop('Attrition_Yes', axis=1)
y = df['Attrition_Yes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Global variables for saved models
MODELS_DIR = "saved_models"
MODELS_LOADED = False
rf_global = None
logreg_global = None
kmeans_global = None
scaler_global = None

def create_models_directory():
    """Create directory for saving models if it doesn't exist"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"✅ Created directory: {MODELS_DIR}")

def save_models(rf, logreg, kmeans, scaler):
    """Save trained models to disk"""
    create_models_directory()
    
    joblib.dump(rf, f"{MODELS_DIR}/random_forest_model.pkl")
    joblib.dump(logreg, f"{MODELS_DIR}/logistic_regression_model.pkl")
    joblib.dump(kmeans, f"{MODELS_DIR}/kmeans_model.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler_model.pkl")
    
    # Save feature names and training info

    training_info = {
        'feature_names': X.columns.tolist(),
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

    joblib.dump(training_info, f"{MODELS_DIR}/training_info.pkl")
    
    print("✅ Models saved successfully!")
    print(f"   - Random Forest: {MODELS_DIR}/random_forest_model.pkl")
    print(f"   - Logistic Regression: {MODELS_DIR}/logistic_regression_model.pkl")
    print(f"   - K-Means: {MODELS_DIR}/kmeans_model.pkl")
    print(f"   - Scaler: {MODELS_DIR}/scaler_model.pkl")
    print(f"   - Training Info: {MODELS_DIR}/training_info.pkl")

def load_models():
    """Load models from disk"""
    global rf_global, logreg_global, kmeans_global, scaler_global, MODELS_LOADED
    
    try:
        print("Loading saved models...")
        rf_global = joblib.load(f"{MODELS_DIR}/random_forest_model.pkl")
        logreg_global = joblib.load(f"{MODELS_DIR}/logistic_regression_model.pkl")
        kmeans_global = joblib.load(f"{MODELS_DIR}/kmeans_model.pkl")
        scaler_global = joblib.load(f"{MODELS_DIR}/scaler_model.pkl")
        training_info = joblib.load(f"{MODELS_DIR}/training_info.pkl")
        
        MODELS_LOADED = True
        print("✅ Models loaded successfully!")
        print(f"   - Features: {len(training_info['feature_names'])}")
        print(f"   - Training samples: {training_info['train_size']}")
        print(f"   - Test samples: {training_info['test_size']}")
        
        return True, training_info
    except FileNotFoundError:
        print("❌ No saved models found. Will train new ones.")
        return False, None

def train_and_save_models():
    """Train all models and save them"""
    global rf_global, logreg_global, kmeans_global, scaler_global, MODELS_LOADED
    
    print("Training all models from scratch...")
    print("=" * 50)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_global = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_global.fit(X_train, y_train)
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    logreg_global = LogisticRegression(max_iter=1000, random_state=42)
    logreg_global.fit(X_train, y_train)
    
    # Train K-Means and Scaler
    print("Training K-Means and Scaler...")
    scaler_global = StandardScaler()
    X_scaled = scaler_global.fit_transform(X)
    kmeans_global = KMeans(n_clusters=2, random_state=42)
    kmeans_global.fit(X_scaled)
    
    # Save all models
    save_models(rf_global, logreg_global, kmeans_global, scaler_global)
    MODELS_LOADED = True
    
    return rf_global, logreg_global, kmeans_global, scaler_global

def get_models():
    """Get models (load from disk or train new ones)"""
    global rf_global, logreg_global, kmeans_global, scaler_global, MODELS_LOADED
    
    if MODELS_LOADED:
        return rf_global, logreg_global, kmeans_global, scaler_global
    
    # Try to load models first
    loaded, training_info = load_models()
    
    if not loaded:
        # If loading failed, train new models
        rf_global, logreg_global, kmeans_global, scaler_global = train_and_save_models()
    
    return rf_global, logreg_global, kmeans_global, scaler_global

# Test Random Forest Model
def test_random_forest():
    print("=" * 50)
    print("TESTING RANDOM FOREST MODEL")
    print("=" * 50)
    
    # Get or train model
    rf, _, _, _ = get_models()
    
    # Predictions
    y_pred_rf = rf.predict(X_test)
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    # Metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    return rf, y_pred_rf, y_pred_proba_rf

# Test Logistic Regression Model
def test_logistic_regression():
    print("=" * 50)
    print("TESTING LOGISTIC REGRESSION MODEL")
    print("=" * 50)  
    
    # Get or train model
    _, logreg, _, _ = get_models()
    
    # Predictions
    y_pred_lr = logreg.predict(X_test)
    y_pred_proba_lr = logreg.predict_proba(X_test)[:, 1]
    
    # Metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(logreg, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_lr)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Logistic Regression - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    return logreg, y_pred_lr, y_pred_proba_lr

# Test K-Means Clustering
def test_kmeans_clustering():
    print("=" * 50)
    print("TESTING K-MEANS CLUSTERING MODEL")
    print("=" * 50)
    
    # Get or train models
    _, _, kmeans, scaler = get_models()
    
    # Scale the data using saved scaler
    X_scaled = scaler.transform(X)
    cluster_labels = kmeans.predict(X_scaled)
    
    # Clustering metrics
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    # Compare clusters with actual attrition
    ari_score = adjusted_rand_score(y, cluster_labels)
    print(f"Adjusted Rand Index: {ari_score:.4f}")
    
    # Cluster analysis
    df_temp = df.copy()
    df_temp['Cluster'] = cluster_labels
    
    print("\nCluster Distribution:")
    print(df_temp['Cluster'].value_counts())
    
    print("\nAttrition Rate by Cluster:")
    attrition_by_cluster = df_temp.groupby('Cluster')['Attrition_Yes'].agg(['count', 'sum', 'mean'])
    attrition_by_cluster.columns = ['Total_Employees', 'Attrition_Count', 'Attrition_Rate']
    print(attrition_by_cluster)
    
    # Test different numbers of clusters
    print("\nTesting different cluster numbers:")
    inertias = []
    silhouette_scores = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans_test = KMeans(n_clusters=k, random_state=42)
        labels = kmeans_test.fit_predict(X_scaled)
        inertias.append(kmeans_test.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        print(f"k={k}: Inertia={kmeans_test.inertia_:.2f}, Silhouette={silhouette_score(X_scaled, labels):.4f}")
    
    # Plot elbow curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    
    plt.tight_layout()
    plt.show()
    
    return kmeans, cluster_labels, X_scaled

# Compare Models Performance
def compare_models(rf_pred, lr_pred):
    print("=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    
    models_performance = {
        'Model': ['Random Forest', 'Logistic Regression'],
        'Accuracy': [accuracy_score(y_test, rf_pred), accuracy_score(y_test, lr_pred)],
        'Precision': [precision_score(y_test, rf_pred), precision_score(y_test, lr_pred)],
        'Recall': [recall_score(y_test, rf_pred), recall_score(y_test, lr_pred)],
        'F1-Score': [f1_score(y_test, rf_pred), f1_score(y_test, lr_pred)]
    }
    
    comparison_df = pd.DataFrame(models_performance)
    print(comparison_df.round(4))
    
    # Visualization
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    rf_scores = [accuracy_score(y_test, rf_pred), precision_score(y_test, rf_pred), 
                 recall_score(y_test, rf_pred), f1_score(y_test, rf_pred)]
    lr_scores = [accuracy_score(y_test, lr_pred), precision_score(y_test, lr_pred), 
                 recall_score(y_test, lr_pred), f1_score(y_test, lr_pred)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, rf_scores, width, label='Random Forest', alpha=0.8)
    plt.bar(x + width/2, lr_scores, width, label='Logistic Regression', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 1)
    
    for i, (rf_score, lr_score) in enumerate(zip(rf_scores, lr_scores)):
        plt.text(i - width/2, rf_score + 0.01, f'{rf_score:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, lr_score + 0.01, f'{lr_score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Interactive Prediction Function (now uses saved models)
def predict_employee_attrition():
    print("=" * 60)
    print("INTERACTIVE EMPLOYEE ATTRITION PREDICTION")
    print("=" * 60)
    
    # Use saved models (much faster!)
    rf, logreg, _, _ = get_models()
    
    print("Please enter the following employee information:")
    print("-" * 50)
    
    # Get feature names and create input dictionary
    feature_names = X.columns.tolist()
    employee_data = {}
    
    # Common features to ask for (adjust based on your actual features)
    feature_prompts = {
        'Age': "Employee's age (e.g., 25, 30, 45): ",
        'MonthlyIncome': "Monthly income (e.g., 5000, 8000, 12000): ",
        'YearsAtCompany': "Years at company (e.g., 1, 5, 10): ",
        'JobSatisfaction': "Job satisfaction (1=Low, 2=Medium, 3=High, 4=Very High): ",
        'WorkLifeBalance': "Work-life balance (1=Bad, 2=Good, 3=Better, 4=Best): ",
        'OverTime_Yes': "Works overtime? (1=Yes, 0=No): ",
        'BusinessTravel_Travel_Frequently': "Travels frequently for business? (1=Yes, 0=No): ",
        'JobLevel': "Job level (1-5, where 5 is highest): ",
        'StockOptionLevel': "Stock option level (0-3): ",
        'TotalWorkingYears': "Total working years (e.g., 5, 10, 20): ",
        'YearsInCurrentRole': "Years in current role (e.g., 1, 3, 7): ",
        'YearsSinceLastPromotion': "Years since last promotion (e.g., 0, 2, 5): ",
        'DistanceFromHome': "Distance from home in km (e.g., 5, 15, 30): ",
        'EnvironmentSatisfaction': "Environment satisfaction (1=Low, 2=Medium, 3=High, 4=Very High): ",
        'JobInvolvement': "Job involvement (1=Low, 2=Medium, 3=High, 4=Very High): "
    }
    
    # Collect user input for available features
    for feature in feature_names:
        if feature in feature_prompts:
            while True:
                try:
                    value = float(input(feature_prompts[feature]))
                    employee_data[feature] = value
                    break
                except ValueError:
                    print("Please enter a valid number.")
        else:
            # For features not in prompts, use median value from training data
            employee_data[feature] = X_train[feature].median()
    
    # Create DataFrame with employee data
    employee_df = pd.DataFrame([employee_data])
    
    # Make predictions
    rf_prediction = rf.predict(employee_df)[0]
    rf_probability = rf.predict_proba(employee_df)[0]
    
    lr_prediction = logreg.predict(employee_df)[0]
    lr_probability = logreg.predict_proba(employee_df)[0]
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    print(f"\n📊 RANDOM FOREST PREDICTION:")
    print(f"   Will Leave: {'YES' if rf_prediction == 1 else 'NO'}")
    print(f"   Probability of Leaving: {rf_probability[1]:.1%}")
    print(f"   Probability of Staying: {rf_probability[0]:.1%}")
    
    print(f"\n📊 LOGISTIC REGRESSION PREDICTION:")
    print(f"   Will Leave: {'YES' if lr_prediction == 1 else 'NO'}")
    print(f"   Probability of Leaving: {lr_probability[1]:.1%}")
    print(f"   Probability of Staying: {lr_probability[0]:.1%}")
    
    # Consensus prediction
    avg_probability = (rf_probability[1] + lr_probability[1]) / 2
    consensus = "HIGH RISK" if avg_probability > 0.6 else "MEDIUM RISK" if avg_probability > 0.3 else "LOW RISK"
    
    print(f"\n🎯 CONSENSUS PREDICTION:")
    print(f"   Average Probability of Leaving: {avg_probability:.1%}")
    print(f"   Risk Level: {consensus}")
    
    # Risk interpretation
    print(f"\n💡 INTERPRETATION:")
    if avg_probability > 0.7:
        print("   ⚠️  VERY HIGH RISK - Immediate attention needed!")
    elif avg_probability > 0.5:
        print("   ⚠️  HIGH RISK - Consider retention strategies")
    elif avg_probability > 0.3:
        print("   ⚡ MEDIUM RISK - Monitor and engage employee")
    else:
        print("   ✅ LOW RISK - Employee likely to stay")
    
    # Feature importance for this prediction (Random Forest)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_,
        'Employee_Value': [employee_data[feature] for feature in feature_names]
    }).sort_values('Importance', ascending=False).head(5)
    
    print(f"\n📈 TOP 5 FACTORS INFLUENCING THIS PREDICTION:")
    for _, row in feature_importance.iterrows():
        print(f"   • {row['Feature']}: {row['Employee_Value']} (Importance: {row['Importance']:.3f})")
    
    return employee_data, rf_prediction, lr_prediction, avg_probability

# Quick prediction with sample data (now uses saved models)
def quick_sample_predictions():
    print("=" * 60)
    print("SAMPLE EMPLOYEE PREDICTIONS")
    print("=" * 60)
    
    # Use saved models
    rf, logreg, _, _ = get_models()
    
    # Sample employees with different risk profiles
    samples = [
        {
            'Profile': 'Young Professional',
            'Age': 25,
            'MonthlyIncome': 3000,
            'YearsAtCompany': 1,
            'JobSatisfaction': 2,
            'WorkLifeBalance': 2,
            'OverTime_Yes': 1
        },
        {
            'Profile': 'Experienced Employee',
            'Age': 40,
            'MonthlyIncome': 8000,
            'YearsAtCompany': 8,
            'JobSatisfaction': 4,
            'WorkLifeBalance': 3,
            'OverTime_Yes': 0
        },
        {
            'Profile': 'Dissatisfied Worker',
            'Age': 35,
            'MonthlyIncome': 4500,
            'YearsAtCompany': 3,
            'JobSatisfaction': 1,
            'WorkLifeBalance': 1,
            'OverTime_Yes': 1
        }
    ]
    
    for sample in samples:
        profile_name = sample.pop('Profile')
        
        # Fill missing features with median values
        full_sample = {}
        for feature in X.columns:
            if feature in sample:
                full_sample[feature] = sample[feature]
            else:
                full_sample[feature] = X_train[feature].median()
        
        sample_df = pd.DataFrame([full_sample])
        
        rf_prob = rf.predict_proba(sample_df)[0][1]
        lr_prob = logreg.predict_proba(sample_df)[0][1]
        avg_prob = (rf_prob + lr_prob) / 2
        
        risk_level = "HIGH" if avg_prob > 0.5 else "MEDIUM" if avg_prob > 0.3 else "LOW"
        
        print(f"\n👤 {profile_name}:")
        print(f"   Attrition Probability: {avg_prob:.1%}")
        print(f"   Risk Level: {risk_level}")
        print(f"   Key Info: Age={sample['Age']}, Income={sample['MonthlyIncome']}, "
              f"Years={sample['YearsAtCompany']}, Satisfaction={sample['JobSatisfaction']}")

# New function to manage models
def model_management_menu():
    print("=" * 60)
    print("MODEL MANAGEMENT")
    print("=" * 60)
    print("1. Train new models and save them")
    print("2. Load existing models")
    print("3. Check if models exist")
    print("4. Delete existing models")
    print("5. Show model info")
    
    choice = input("Choose an option (1-5): ")
    
    if choice == "1":
        train_and_save_models()
    elif choice == "2":
        loaded, info = load_models()
        if loaded:
            print("Models loaded successfully!")
        else:
            print("No models found to load.")
    elif choice == "3":
        if os.path.exists(MODELS_DIR):
            files = os.listdir(MODELS_DIR)
            print(f"Models directory exists with {len(files)} files:")
            for file in files:
                print(f"   - {file}")
        else:
            print("No models directory found.")
    elif choice == "4":
        if os.path.exists(MODELS_DIR):
            import shutil
            shutil.rmtree(MODELS_DIR)
            global MODELS_LOADED
            MODELS_LOADED = False
            print("✅ All saved models deleted.")
        else:
            print("No models directory to delete.")
    elif choice == "5":
        if MODELS_LOADED:
            print("Current models in memory:")
            print(f"   - Random Forest: {type(rf_global).__name__}")
            print(f"   - Logistic Regression: {type(logreg_global).__name__}")
            print(f"   - K-Means: {type(kmeans_global).__name__}")
            print(f"   - Scaler: {type(scaler_global).__name__}")
        else:
            print("No models currently loaded in memory.")

# Run all tests
if __name__ == "__main__":
    print("=" * 60)
    print("EMPLOYEE ATTRITION PREDICTION SYSTEM")
    print("=" * 60)
    
    print("Choose what you want to do:")
    print("1. Run complete model testing")
    print("2. Interactive prediction only")
    print("3. Model management")
    print("4. Quick sample predictions")
    
    main_choice = input("Enter your choice (1-4): ")
    
    if main_choice == "1":
        print("Starting comprehensive model testing...")
        
        # Test individual models
        rf_model, rf_predictions, rf_probabilities = test_random_forest()
        lr_model, lr_predictions, lr_probabilities = test_logistic_regression()
        kmeans_model, cluster_labels, scaled_data = test_kmeans_clustering()
        
        # Compare supervised models
        compare_models(rf_predictions, lr_predictions)
        
        # After adding cluster or prediction labels
        df_output = df.copy()  # Create a copy to avoid modifying original data
        
        # Scale the full dataset for clustering
        _, _, kmeans, scaler = get_models()
        X_scaled_full = scaler.transform(X)
        cluster_labels_full = kmeans.predict(X_scaled_full)
        
        df_output['Cluster'] = cluster_labels_full  # from KMeans
        df_output['Attrition_Prediction'] = rf_model.predict(X)  # from classifier
        
        # Save to CSV
        df_output.to_csv(r"C:\Users\pc\Desktop\employee_analysis_output.csv", index=False)
        print("✅ Analysis results saved to 'employee_analysis_output.csv'")
        print(f"   - Total records: {len(df_output)}")
        print(f"   - Columns: {list(df_output.columns)}")
        
        # Show sample predictions
        quick_sample_predictions()
        
        # Interactive prediction
        print("\n" + "=" * 60)
        while True:
            choice = input("Do you want to predict attrition for a specific employee? (y/n): ").lower()
            if choice == 'y':
                predict_employee_attrition()
                print("\n")
            else:
                break
        
        print("=" * 50)
        print("TESTING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
    elif main_choice == "2":
        while True:
            predict_employee_attrition()
            choice = input("\nPredict for another employee? (y/n): ").lower()
            if choice != 'y':
                break
                
    elif main_choice == "3":
        model_management_menu()
        
    elif main_choice == "4":
        quick_sample_predictions()



