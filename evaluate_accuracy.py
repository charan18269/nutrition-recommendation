import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Define the model class (same as in train_model.py)
class NutritionRecommendationModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoders = {}
        
    def preprocess_data(self, df):
        # Create copy of dataframe
        processed_df = df.copy()
        
        # Convert timestamp to datetime features
        processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
        processed_df['day_of_week'] = processed_df['timestamp'].dt.dayofweek
        processed_df['month'] = processed_df['timestamp'].dt.month
        
        # Drop original timestamp column
        processed_df = processed_df.drop('timestamp', axis=1)
        
        # Encode categorical variables
        categorical_columns = ['user_gender', 'craving', 'primary_deficiency', 
                             'secondary_deficiency', 'recommended_food']
        
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                processed_df[column] = self.label_encoders[column].fit_transform(processed_df[column].fillna('None'))
            else:
                processed_df[column] = self.label_encoders[column].transform(processed_df[column].fillna('None'))
        
        return processed_df
    
    def predict(self, X):
        return self.model.predict(X)

def evaluate_model():
    # First check if model exists
    try:
        # Load the model
        print("Loading model...")
        model = joblib.load('nutrition_model.joblib')
    except FileNotFoundError:
        print("Error: Model file not found. Please run train_model.py first.")
        return
    
    # Load the test data
    print("Loading test data...")
    try:
        df = pd.read_csv('nutrition_recommendation_data.csv')
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return
    
    # Preprocess the test data
    processed_df = model.preprocess_data(df)
    
    # Separate features and target
    X = processed_df.drop(['success', 'satisfaction_score'], axis=1)
    y = processed_df['success']
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X)
    
    # Calculate detailed metrics
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate accuracy for different user segments
    print("\nAccuracy by User Segments:")
    
    # Gender-based accuracy
    gender_accuracy = {}
    for gender_code in df['user_gender'].unique():
        mask = df['user_gender'] == gender_code
        gender_pred = y_pred[mask]
        gender_true = y[mask]
        gender_accuracy[gender_code] = (gender_pred == gender_true).mean()
    
    print("\nAccuracy by Gender:")
    for gender, acc in gender_accuracy.items():
        print(f"Gender {gender}: {acc:.2%}")
    
    # Age group accuracy
    df['age_group'] = pd.cut(df['user_age'], bins=[0, 25, 35, 50, 100], 
                            labels=['18-25', '26-35', '36-50', '50+'])
    
    age_accuracy = {}
    for age_group in df['age_group'].unique():
        mask = df['age_group'] == age_group
        age_pred = y_pred[mask]
        age_true = y[mask]
        age_accuracy[age_group] = (age_pred == age_true).mean()
    
    print("\nAccuracy by Age Group:")
    for age_group, acc in age_accuracy.items():
        print(f"{age_group}: {acc:.2%}")
    
    # Save detailed results to file
    with open('model_evaluation_results.txt', 'w') as f:
        f.write("Classification Report:\n")
        f.write(classification_report(y, y_pred))
        f.write("\n\nAccuracy by Gender:\n")
        for gender, acc in gender_accuracy.items():
            f.write(f"Gender {gender}: {acc:.2%}\n")
        f.write("\nAccuracy by Age Group:\n")
        for age_group, acc in age_accuracy.items():
            f.write(f"{age_group}: {acc:.2%}\n")

if __name__ == "__main__":
    evaluate_model() 