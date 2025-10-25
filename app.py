import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb


@st.cache_data
def load_assets():
    """Loads the model and original training data."""
    model = xgb.XGBRegressor()
    model.load_model("model.json")
    

    train_df = pd.read_csv("original_train_for_app.csv")
    target = train_df.columns.tolist()[-1] # Assumes target is last column
    
    return model, train_df, target


@st.cache_data
def get_unique_values(train_df):
    """Gets unique values for dropdowns, handling potential errors."""
    unique_values = {}
    
    categorical_cols = ['lighting', 'weather', 'road_type', 'num_lanes', 'road_signs_present', 'time_of_day', 'public_road', 'holiday', 'school_season']
    

    for col in categorical_cols:
        try:
            unique_values[col] = sorted(train_df[col].unique().tolist())
        except KeyError:
            st.error(f"Error: Column '{col}' not found in original_train_for_app.csv. Please check the file.")
            if col == 'lighting': unique_values[col] = ['daylight', 'night']
            elif col == 'weather': unique_values[col] = ['clear', 'rainy']
            else: unique_values[col] = ['Option 1', 'Option 2']

    return unique_values


def create_frequency_features(train_df, test_df, cols, num, cat):
    train, test = train_df.copy(), test_df.copy()
    target = train.columns.tolist()[-1] # Get target name

    for col in cols:
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq)
        test[f"{col}_freq"] = test[col].map(freq).fillna(train[f"{col}_freq"].mean())

        if col in num:
            for q in [5, 10, 15]:
                try:
                    train[f"{col}_bin{q}"], bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                    test[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True).fillna(0) # Fill NaNs
                except Exception:
                    train[f"{col}_bin{q}"] = test[f"{col}_bin{q}"] = 0

    return train, test


def add_new_features(train_df, test_df, orig_df=None):
    train, test = train_df.copy(), test_df.copy()
    
    # 1. Interaction Feature
    train['speed_curvature_interact'] = train['speed_limit'] * train['curvature']
    test['speed_curvature_interact'] = test['speed_limit'] * test['curvature']

    # 2. Binary Indicator
    train['high_speed'] = (train['speed_limit'] >= 60).astype(int)
    test['high_speed'] = (test['speed_limit'] >= 60).astype(int)

    # 3. Risk Score
    train['risk_score'] = (
        0.3 * train['curvature'] +
        0.2 * (train['lighting'] == 'night').astype(int) +
        0.1 * (train['weather'] != 'clear').astype(int) +
        0.2 * train['high_speed'] +
        0.1 * (train['num_reported_accidents'] > 2).astype(int)
    )
    test['risk_score'] = (
        0.3 * test['curvature'] +
        0.2 * (test['lighting'] == 'night').astype(int) +
        0.1 * (test['weather'] != 'clear').astype(int) +
        0.2 * test['high_speed'] +
        0.1 * (test['num_reported_accidents'] > 2).astype(int)
    )
    
    return train, test


def run_prediction_pipeline(input_df, train_df, target):
    """
    Runs the full preprocessing pipeline on new user input,
    using the original train_df for statistics.
    """

    train_copy = train_df.copy()
    test_copy = input_df.copy()
    
    if 'id' not in train_copy.columns:
        train_copy['id'] = range(len(train_copy))
    if 'id' not in test_copy.columns:
        test_copy['id'] = [0] 

    cols = train_copy.drop(columns=[target, 'id']).columns.tolist()
    cat = [col for col in cols if train_copy[col].dtype in ["object","category"] and col != target]
    num = [col for col in cols if train_copy[col].dtype not in ["object","category","bool"] and col not in ["id", target]]

    _, test_copy = create_frequency_features(train_copy, test_copy, cols, num, cat)


    for col in cat:
        train_cats = train_copy[col].astype('category').cat.categories
        test_copy[col] = pd.Categorical(test_copy[col], categories=train_cats)

    map_col = "num_reported_accidents"
    map_num_reported = {0:0, 1:0, 2:0, 3:2, 4:4, 5:3, 6:1, 7:0}
    test_copy[map_col] = test_copy[map_col].map(map_num_reported)

    remove = ["time_of_day", "num_lanes", "road_type", "road_signs_present", "id_freq"]
    cols_to_drop = [col for col in remove if col in test_copy.columns]
    test_copy = test_copy.drop(columns=cols_to_drop)

    _, test_copy = add_new_features(train_copy, test_copy)
    
    if 'id' in test_copy.columns:
        test_copy = test_copy.drop(columns="id")
    
    return test_copy


# --- Streamlit App ---
st.set_page_config(page_title="Road Risk Predictor", layout="wide")
st.title("🚧 Road Accident Risk Predictor 🚗")
st.write("Based on the Kaggle Playground Series (S5E10). Use the controls on the left to describe a road, and the model will predict its accident risk score.")

# Load model and data
try:
    model, train_df, target = load_assets()
    unique_values = get_unique_values(train_df)
except FileNotFoundError as e:
    st.error(f"Error: Missing asset file ({e.fileName}).")
    st.error("Please make sure 'model.xgb' and 'original_train_for_app.csv' are in the same folder as 'app.py'.")
    st.stop()


# User Inputs in Sidebar
st.sidebar.header("Select Road Conditions")


speed_min, speed_max = int(train_df['speed_limit'].min()), int(train_df['speed_limit'].max())
curve_min, curve_max = float(train_df['curvature'].min()), float(train_df['curvature'].max())
acc_min, acc_max = int(train_df['num_reported_accidents'].min()), int(train_df['num_reported_accidents'].max())


# Create the UI elements
speed_limit = st.sidebar.slider("Speed Limit (km/h)", speed_min, speed_max, 60)
curvature = st.sidebar.slider("Road Curvature", curve_min, curve_max, 0.2, 0.01)
num_reported_accidents = st.sidebar.slider("Reported Accidents (in area)", acc_min, acc_max, 2)

lighting = st.sidebar.selectbox("Lighting Conditions", unique_values['lighting'])
weather = st.sidebar.selectbox("Weather Conditions", unique_values['weather'])
road_type = st.sidebar.selectbox("Road Type", unique_values['road_type'])
num_lanes = st.sidebar.selectbox("Number of Lanes", unique_values['num_lanes'])
road_signs_present = st.sidebar.selectbox("Road Signs Present", unique_values['road_signs_present'])
time_of_day = st.sidebar.selectbox("Time of Day", unique_values['time_of_day'])
public_road = st.sidebar.selectbox("Public Road", unique_values['public_road'])
holiday = st.sidebar.selectbox("Holiday", unique_values['holiday'])
school_season = st.sidebar.selectbox("School Season", unique_values['school_season'])


if st.sidebar.button("Predict Accident Risk"):
    input_data = {
        'id': [0],
        'speed_limit': [speed_limit],
        'public_road': [public_road],
        'lighting': [lighting],
        'weather': [weather],
        'road_type': [road_type],
        'num_lanes': [num_lanes],
        'curvature': [curvature],
        'num_reported_accidents': [num_reported_accidents],
        'road_signs_present': [road_signs_present],
        'time_of_day': [time_of_day], 
        'holiday': [holiday],
        'school_season': [school_season]
    }
    input_df = pd.DataFrame.from_dict(input_data)
    
    processed_input = run_prediction_pipeline(input_df, train_df, target)

    processed_input = processed_input[model.get_booster().feature_names]
    

    try:
        prediction = model.predict(processed_input)[0]
    
        st.header("Prediction Result")
        st.metric(label="Predicted Accident Risk Score", value=f"{prediction:.4f}")

        st.progress(float(prediction))  # Cast to float; assumes score is 0-1

        st.info("This score is a prediction from the XGBoost model. A higher score indicates a higher predicted risk of an accident.", icon="ℹ️")
        
        with st.expander("Show Processed Features Sent to Model"):
            st.dataframe(processed_input)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("This might be due to a mismatch in expected features. Check the pipeline.")
        st.dataframe(processed_input) 