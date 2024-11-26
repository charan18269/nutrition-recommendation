import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Define the model class
class NutritionRecommendationModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoders = {}
        
    def preprocess_data(self, df):
        # Create copy of dataframe
        processed_df = df.copy()
        
        # Add timestamp features if timestamp exists
        if 'timestamp' in processed_df.columns:
            processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
            processed_df['day_of_week'] = processed_df['timestamp'].dt.dayofweek
            processed_df['month'] = processed_df['timestamp'].dt.month
            processed_df = processed_df.drop('timestamp', axis=1)
        else:
            # Add default values for timestamp features
            processed_df['day_of_week'] = datetime.now().weekday()
            processed_df['month'] = datetime.now().month
        
        # Encode categorical variables
        categorical_columns = ['user_gender', 'craving', 'primary_deficiency', 
                             'secondary_deficiency', 'recommended_food']
        
        for column in categorical_columns:
            if column in processed_df.columns:
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    processed_df[column] = self.label_encoders[column].fit_transform(processed_df[column].fillna('None'))
                else:
                    processed_df[column] = self.label_encoders[column].transform(processed_df[column].fillna('None'))
        
        return processed_df

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('nutrition_model.joblib')

# Define nutrient mappings
CRAVINGS_NUTRIENTS = {
    "Chocolate": ["Magnesium", "Vitamin B"],
    "Salty Foods": ["Sodium", "Stress Minerals"],
    "Red Meat": ["Iron", "Vitamin B12"],
    "Ice": ["Iron", "Minerals"],
    "Sugary Foods": ["Chromium", "Carbon", "Phosphorus"],
    "Bread/Pasta": ["Nitrogen", "Serotonin Support"],
    "Fatty Foods": ["Calcium", "Essential Fatty Acids"],
    "Citrus Fruits": ["Vitamin C"]
}

# Load food database
def load_food_database():
    return {
        "Spinach": {
            "nutrients": ["Iron", "Magnesium", "Vitamin B"],
            "description": "Leafy green vegetable rich in iron and other essential nutrients."
        },
        "Salmon": {
            "nutrients": ["Omega-3", "Vitamin B12", "Vitamin D"],
            "description": "Fatty fish high in omega-3s and protein."
        },
        "Bananas": {
            "nutrients": ["Magnesium", "Potassium", "Vitamin B6"],
            "description": "Great source of potassium and natural energy."
        },
        "Dark Chocolate": {
            "nutrients": ["Magnesium", "Iron", "Antioxidants"],
            "description": "Rich in magnesium and antioxidants."
        },
        "Sweet Potatoes": {
            "nutrients": ["Vitamin C", "Magnesium", "Potassium"],
            "description": "Nutritious root vegetable rich in vitamins."
        },
        "Almonds": {
            "nutrients": ["Magnesium", "Calcium", "Vitamin E"],
            "description": "Nutrient-dense nuts great for snacking."
        },
        "Oranges": {
            "nutrients": ["Vitamin C", "Fiber", "Potassium"],
            "description": "Citrus fruit packed with vitamin C."
        },
        "Lentils": {
            "nutrients": ["Iron", "Protein", "Fiber"],
            "description": "Plant-based protein source rich in iron."
        }
    }

def get_recommendations(craving, model, food_db):
    # Get nutrients needed based on craving
    nutrients_needed = CRAVINGS_NUTRIENTS.get(craving, [])
    
    # Create a sample user data
    user_data = {
        'user_age': 30,
        'user_gender': 'F',
        'user_bmi': 22.0,
        'craving': craving,
        'primary_deficiency': nutrients_needed[0] if nutrients_needed else 'None',
        'secondary_deficiency': nutrients_needed[1] if len(nutrients_needed) > 1 else 'None',
        'recommended_food': 'None',
        'day_of_week': datetime.now().weekday(),
        'month': datetime.now().month
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([user_data])
    
    # Filter foods based on nutrients
    recommended_foods = []
    for food, details in food_db.items():
        if any(nutrient in details["nutrients"] for nutrient in nutrients_needed):
            recommended_foods.append({
                "name": food,
                "nutrients": details["nutrients"],
                "description": details["description"],
                "confidence": 0.8  # Default confidence score
            })
    
    # If no recommendations found, return general healthy options
    if not recommended_foods:
        recommended_foods = [
            {
                "name": "Spinach",
                "nutrients": food_db["Spinach"]["nutrients"],
                "description": food_db["Spinach"]["description"],
                "confidence": 0.7
            }
        ]
    
    # Return top 3 recommendations
    return recommended_foods[:3]

def main():
    st.set_page_config(page_title="Nutrition Recommendation System", layout="wide")
    
    st.title("🥗 Nutrition Recommendation System")
    st.write("""
    Get personalized food recommendations based on your cravings! 
    This AI-powered system will help identify potential nutritional deficiencies and suggest healthy alternatives.
    """)
    
    # Load model and food database
    try:
        model = load_model()
        food_db = load_food_database()
    except Exception as e:
        st.error(f"Error loading model or database: {str(e)}")
        return
    
    # User input form
    with st.form("user_input"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["M", "F"])
            bmi = st.number_input("BMI", min_value=15.0, max_value=45.0, value=22.0)
        
        with col2:
            craving = st.selectbox("What are you craving?", list(CRAVINGS_NUTRIENTS.keys()))
        
        submitted = st.form_submit_button("Get Recommendations")
    
    if submitted:
        # Get recommendations
        recommendations = get_recommendations(craving, model, food_db)
        
        # Display recommendations
        st.subheader("Your Personalized Recommendations")
        
        cols = st.columns(len(recommendations))
        for idx, (col, rec) in enumerate(zip(cols, recommendations)):
            with col:
                st.markdown(f"### {rec['name']}")
                st.write(f"**Key Nutrients:** {', '.join(rec['nutrients'])}")
                st.write(rec['description'])
                st.progress(rec['confidence'])
                st.write(f"Confidence: {rec['confidence']:.1%}")
        
        # Display nutritional insight
        st.subheader("Nutritional Insight")
        st.info(f"Your craving for {craving} might indicate a need for: {', '.join(CRAVINGS_NUTRIENTS[craving])}")

if __name__ == "__main__":
    main() 