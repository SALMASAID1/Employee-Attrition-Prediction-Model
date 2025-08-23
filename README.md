# Employee Attrition Prediction Project

This project aims to predict employee attrition using various machine learning models. It includes data preprocessing, model training, and a Streamlit application for user interaction.

## Project Structure

```
employee-attrition-prediction/
├── .venv/                         # Virtual environment (keep it here)
├── saved_models/                  # Trained models (.pkl files)
│   ├── random_forest_model.pkl    # Trained Random Forest model
│   ├── logistic_regression_model.pkl # Trained Logistic Regression model
│   ├── kmeans_model.pkl           # Trained K-Means clustering model
│   ├── scaler_model.pkl           # Scaler model for normalizing input features
│   └── training_info.pkl          # Metadata about the training process
│
├── data/                          # Raw and processed datasets
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Original dataset
│   ├── cleaned_employee_attrition.csv          # Cleaned dataset
│   ├── encoded_employee_attrition.csv          # Encoded dataset
│   └── employee_analysis_output.csv            # Predictions + clusters
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── datavisualization.ipynb    # Data visualization and exploratory analysis
│   └── modelstest.ipynb           # Testing different models and evaluating performance
│
├── src/                           # Source Python scripts
│   ├── save_testmodels.py         # Script to train & save models
│   └── utils.py                   # Helper functions (scaling, metrics, etc.)
│
├── app/                           # Streamlit app
│   └── app.py                     # Main Streamlit dashboard
│
├── requirements.txt               # Python dependencies
├── .gitignore                     # Ignore venv, data, models (if needed for GitHub)
└── README.md                      # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd employee-attrition-prediction
   ```

2. Create a virtual environment:
   ```
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source .venv/bin/activate
     ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. To train the models, run the `save_testmodels.py` script in the `src` directory.
2. Use the Jupyter notebooks in the `notebooks` directory for data exploration and model testing.
3. Launch the Streamlit app:
   ```
   streamlit run app/app.py
   ```
   ##Author
   
   SALMA SAID
