import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

@st.cache_data
def load_data():
    model = xgb.XGBRegressor()
    model.load_model("model.json")

    df = pd.read_csv("original_train_for_app.csv")
    target = df.columns[-1]
    return model, df, target


@st.cache_data
def get_unique(df):
    cols = ['lighting', 'weather', 'road_type', 'num_lanes',
            'road_signs_present', 'time_of_day', 'public_road',
            'holiday', 'school_season']
    uniq = {}
    for c in cols:
        try:
            uniq[c] = sorted(df[c].unique().tolist())
        except KeyError:
            uniq[c] = ['Option 1', 'Option 2']
    return uniq


def freq_features(train, test, cols, num, cat):
    train, test = train.copy(), test.copy()
    target = train.columns[-1]
    for c in cols:
        freq = train[c].value_counts(normalize=True)
        train[f"{c}_freq"] = train[c].map(freq)
        test[f"{c}_freq"] = test[c].map(freq).fillna(train[f"{c}_freq"].mean())

        if c in num:
            for q in [5, 10, 15]:
                try:
                    train[f"{c}_bin{q}"], bins = pd.qcut(train[c], q=q, labels=False, retbins=True, duplicates="drop")
                    test[f"{c}_bin{q}"] = pd.cut(test[c], bins=bins, labels=False, include_lowest=True).fillna(0)
                except Exception:
                    train[f"{c}_bin{q}"] = test[f"{c}_bin{q}"] = 0
    return train, test


def add_features(train, test):
    train, test = train.copy(), test.copy()

    train['speed_curvature'] = train['speed_limit'] * train['curvature']
    test['speed_curvature'] = test['speed_limit'] * test['curvature']

    train['high_speed'] = (train['speed_limit'] >= 60).astype(int)
    test['high_speed'] = (test['speed_limit'] >= 60).astype(int)

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


def prepare_input(input_df, train_df, target):
    train, test = train_df.copy(), input_df.copy()

    if 'id' not in train: train['id'] = range(len(train))
    if 'id' not in test: test['id'] = [0]

    cols = train.drop(columns=[target, 'id']).columns.tolist()
    cat = [c for c in cols if train[c].dtype == 'object']
    num = [c for c in cols if c not in cat + ['id', target]]

    _, test = freq_features(train, test, cols, num, cat)

    for c in cat:
        cats = train[c].astype('category').cat.categories
        test[c] = pd.Categorical(test[c], categories=cats)

    if "num_reported_accidents" in test:
        mapping = {0: 0, 1: 0, 2: 0, 3: 2, 4: 4, 5: 3, 6: 1, 7: 0}
        test["num_reported_accidents"] = test["num_reported_accidents"].map(mapping)

    drop_cols = [c for c in ["time_of_day", "num_lanes", "road_type", "road_signs_present", "id_freq"] if c in test]
    test = test.drop(columns=drop_cols, errors='ignore')

    _, test = add_features(train, test)

    if 'id' in test:
        test = test.drop(columns='id')

    return test


# --- Streamlit UI ---

st.set_page_config(page_title="Road Risk Predictor", layout="wide")
st.title("🚧 Road Accident Risk Predictor")
st.write("Describe the road conditions to estimate the accident risk score.")

try:
    model, df, target = load_data()
    uniq = get_unique(df)
except FileNotFoundError:
    st.error("Missing model or CSV file.")
    st.stop()

st.sidebar.header("Road Conditions")

speed_min, speed_max = int(df['speed_limit'].min()), int(df['speed_limit'].max())
curve_min, curve_max = float(df['curvature'].min()), float(df['curvature'].max())
acc_min, acc_max = int(df['num_reported_accidents'].min()), int(df['num_reported_accidents'].max())

speed = st.sidebar.slider("Speed Limit (km/h)", speed_min, speed_max, 60)
curv = st.sidebar.slider("Curvature", curve_min, curve_max, 0.2, 0.01)
acc = st.sidebar.slider("Reported Accidents", acc_min, acc_max, 2)

lighting = st.sidebar.selectbox("Lighting", uniq['lighting'])
weather = st.sidebar.selectbox("Weather", uniq['weather'])
road_type = st.sidebar.selectbox("Road Type", uniq['road_type'])
lanes = st.sidebar.selectbox("Lanes", uniq['num_lanes'])
signs = st.sidebar.selectbox("Road Signs", uniq['road_signs_present'])
time = st.sidebar.selectbox("Time of Day", uniq['time_of_day'])
public = st.sidebar.selectbox("Public Road", uniq['public_road'])
holiday = st.sidebar.selectbox("Holiday", uniq['holiday'])
school = st.sidebar.selectbox("School Season", uniq['school_season'])

if st.sidebar.button("Predict"):
    data = {
        'id': [0],
        'speed_limit': [speed],
        'public_road': [public],
        'lighting': [lighting],
        'weather': [weather],
        'road_type': [road_type],
        'num_lanes': [lanes],
        'curvature': [curv],
        'num_reported_accidents': [acc],
        'road_signs_present': [signs],
        'time_of_day': [time],
        'holiday': [holiday],
        'school_season': [school]
    }

    input_df = pd.DataFrame(data)
    processed = prepare_input(input_df, df, target)
    processed = processed[model.get_booster().feature_names]

    try:
        pred = model.predict(processed)[0]
        st.header("Predicted Risk Score")
        st.metric("Accident Risk", f"{pred:.4f}")
        st.progress(float(pred))
        st.info("Higher score = higher predicted risk.")
        with st.expander("Show processed features"):
            st.dataframe(processed)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.dataframe(processed)
