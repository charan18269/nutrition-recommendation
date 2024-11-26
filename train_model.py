import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

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
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names):
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))

def main():
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('nutrition_recommendation_data.csv')
    
    # Initialize model
    model = NutritionRecommendationModel()
    
    # Preprocess data
    print("Preprocessing data...")
    processed_df = model.preprocess_data(df)
    
    # Separate features and target
    X = processed_df.drop(['success', 'satisfaction_score'], axis=1)
    y = processed_df['success']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print("Training model...")
    model.train(X_train, y_train)
    
    # Calculate and print accuracy
    train_accuracy = model.model.score(X_train, y_train)
    test_accuracy = model.model.score(X_test, y_test)
    
    print(f"\nTraining Accuracy: {train_accuracy:.2%}")
    print(f"Testing Accuracy: {test_accuracy:.2%}")
    
    # Print feature importance
    print("\nFeature Importance:")
    importance = model.get_feature_importance(X.columns)
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {imp:.4f}")
    
    # Save the model and label encoders
    print("\nSaving model and encoders...")
    joblib.dump(model, 'nutrition_model.joblib')
    print("Model saved successfully!")

if __name__ == "__main__":
    main() 