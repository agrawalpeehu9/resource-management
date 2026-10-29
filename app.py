# AI for Sustainable Resource Management - Streamlit app
# Student: Peehu Agrawal | Reg. no: 2428010207
# Run:  pip install streamlit scikit-learn pandas numpy matplotlib
#       streamlit run app.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

st.title("AI for Sustainable Resource Management")
st.write("Predicts the normal daily use of electricity and water, and finds days with wastage.")


# ---------- Step 1: create the dataset ----------
@st.cache_data
def make_data():
    np.random.seed(7)
    df = pd.DataFrame()
    df['date'] = pd.date_range('2025-01-01', '2025-12-31')
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)

    df['holiday'] = 0
    df.loc[np.random.choice(365, 12, replace=False), 'holiday'] = 1

    month_temp = [15, 18, 24, 30, 35, 36, 32, 31, 30, 27, 21, 16]
    df['temp'] = df['month'].apply(lambda m: month_temp[m - 1]) + np.random.normal(0, 1.5, 365)

    df['occupancy'] = 900
    df.loc[(df['weekend'] == 1) | (df['holiday'] == 1), 'occupancy'] = 250
    df.loc[df['month'].isin([6, 7]), 'occupancy'] = df['occupancy'] * 0.5
    df['occupancy'] = df['occupancy'] + np.random.normal(0, 40, 365)

    df['electricity'] = 300 + 0.9 * df['occupancy'] + 18 * (df['temp'] - 24).clip(lower=0) + np.random.normal(0, 25, 365)
    df['water'] = 20 + 0.08 * df['occupancy'] + 1.5 * (df['temp'] - 20).clip(lower=0) + np.random.normal(0, 4, 365)

    df['wastage'] = 0
    waste_days = np.random.choice(365, 40, replace=False)
    df.loc[waste_days, 'wastage'] = 1
    df.loc[waste_days, 'electricity'] = df.loc[waste_days, 'electricity'] * np.random.uniform(1.1, 1.4, 40)
    df.loc[waste_days, 'water'] = df.loc[waste_days, 'water'] * np.random.uniform(1.1, 1.4, 40)
    return df


df = make_data()
features = ['month', 'day_of_week', 'weekend', 'holiday', 'temp', 'occupancy']

# ---------- Step 2: Part A - regression ----------
normal = df[df['wastage'] == 0]
X_train, X_test, ye_train, ye_test, yw_train, yw_test = train_test_split(
    normal[features], normal['electricity'], normal['water'], test_size=0.2, random_state=42)

lr_elec = LinearRegression().fit(X_train, ye_train)
rf_elec = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, ye_train)
lr_water = LinearRegression().fit(X_train, yw_train)
rf_water = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, yw_train)


def reg_scores(actual, predicted):
    return [round(mean_absolute_error(actual, predicted), 2),
            round(mean_squared_error(actual, predicted), 2)]


elec_results = pd.DataFrame(
    [reg_scores(ye_test, lr_elec.predict(X_test)), reg_scores(ye_test, rf_elec.predict(X_test))],
    index=['Linear Regression', 'Random Forest'], columns=['MAE', 'MSE'])
water_results = pd.DataFrame(
    [reg_scores(yw_test, lr_water.predict(X_test)), reg_scores(yw_test, rf_water.predict(X_test))],
    index=['Linear Regression', 'Random Forest'], columns=['MAE', 'MSE'])

# ---------- Step 3: Part B - classification ----------
df['pred_elec'] = rf_elec.predict(df[features])
df['pred_water'] = rf_water.predict(df[features])
df['elec_gap'] = (df['electricity'] - df['pred_elec']) / df['pred_elec']
df['water_gap'] = (df['water'] - df['pred_water']) / df['pred_water']

X2 = df[['elec_gap', 'water_gap']]
y2 = df['wastage']
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2)

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}
rows = []
for name, model in models.items():
    model.fit(X2_train, y2_train)
    pred = model.predict(X2_test)
    cv = cross_val_score(model, X2, y2, cv=5, scoring='f1').mean()
    rows.append([round(accuracy_score(y2_test, pred), 3), round(f1_score(y2_test, pred), 3), round(cv, 3)])
class_results = pd.DataFrame(rows, index=list(models.keys()),
                             columns=['Accuracy', 'F1 score', 'CV F1 (5-fold)'])

# ---------- Show everything on the page ----------
tab1, tab2, tab3, tab4 = st.tabs(["1. Dataset", "2. Part A: Regression", "3. Part B: Classification", "4. Wastage days"])

with tab1:
    st.subheader("Daily data for 2025 (simulated)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Days", len(df))
    c2.metric("Normal days", int((df['wastage'] == 0).sum()))
    c3.metric("Wastage days", int(df['wastage'].sum()))
    st.dataframe(df[['date'] + features + ['electricity', 'water', 'wastage']].head(10))
    st.write("Summary statistics")
    st.dataframe(df[['electricity', 'water', 'temp', 'occupancy']].describe().round(2))

    monthly = df.groupby('month')[['electricity', 'water']].sum()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].bar(monthly.index, monthly['electricity'] / 1000, color='steelblue')
    ax[0].set_title('Monthly electricity (MWh)')
    ax[1].bar(monthly.index, monthly['water'] / 1000, color='seagreen')
    ax[1].set_title('Monthly water (thousand kL)')
    st.pyplot(fig)

with tab2:
    st.subheader("Predicting normal use (Linear Regression vs Random Forest)")
    st.write("MAE = average error, MSE = average squared error. Lower is better.")
    st.write("Electricity (kWh)")
    st.dataframe(elec_results)
    st.write("Water (kL)")
    st.dataframe(water_results)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    for i, metric in enumerate(['MAE', 'MSE']):
        ax[i].bar(['Linear Reg.', 'Random Forest'], elec_results[metric], color=['orange', 'seagreen'])
        ax[i].set_title('Electricity ' + metric)
    st.pyplot(fig)

    st.subheader("Try it yourself")
    month = st.slider("Month", 1, 12, 5)
    temp = st.slider("Temperature (°C)", 5, 45, 32)
    people = st.slider("People present", 0, 1200, 900)
    day = st.selectbox("Day of the week", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    dow = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].index(day)
    holiday = st.checkbox("Holiday")
    row = pd.DataFrame([[month, dow, int(dow >= 5), int(holiday), temp, people]], columns=features)
    st.write("Expected electricity:", round(rf_elec.predict(row)[0]), "kWh")
    st.write("Expected water:", round(rf_water.predict(row)[0], 1), "kL")

with tab3:
    st.subheader("Is a day a wastage day?")
    st.write("Gap = (actual use - expected use) / expected use. The gaps are the inputs.")
    st.dataframe(class_results)
    best_name = class_results['F1 score'].idxmax()
    st.write("Best classifier:", best_name)

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    for i, metric in enumerate(['Accuracy', 'F1 score']):
        ax[i].bar(['Log. Reg.', 'Dec. Tree', 'KNN'], class_results[metric], color=['orange', 'steelblue', 'seagreen'])
        ax[i].set_title(metric)
        ax[i].set_ylim(0, 1.05)
    st.pyplot(fig)

    cm = confusion_matrix(y2_test, models[best_name].predict(X2_test))
    st.write("Confusion matrix (rows = actual, columns = predicted)")
    st.dataframe(pd.DataFrame(cm, index=['Actual normal', 'Actual wastage'],
                              columns=['Predicted normal', 'Predicted wastage']))

with tab4:
    best_model = models[best_name]
    df['flag'] = best_model.predict(df[['elec_gap', 'water_gap']])
    extra = (df['electricity'] - df['pred_elec'])[df['flag'] == 1].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Days flagged", int(df['flag'].sum()))
    c2.metric("Extra electricity (kWh)", round(extra))
    c3.metric("Share of yearly use", str(round(100 * extra / df['electricity'].sum(), 2)) + " %")

    part = df[(df['date'] >= '2025-03-01') & (df['date'] < '2025-04-15')]
    flagged = part[part['flag'] == 1]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(part['date'], part['electricity'], label='Actual', color='steelblue')
    ax.plot(part['date'], part['pred_elec'], label='Expected', color='orange', linestyle='--')
    ax.scatter(flagged['date'], flagged['electricity'], color='red', label='Wastage flagged', zorder=3)
    ax.set_ylabel('Electricity (kWh)')
    ax.legend()
    st.pyplot(fig)

    st.write("Flagged days")
    st.dataframe(df[df['flag'] == 1][['date', 'electricity', 'pred_elec', 'water', 'pred_water']].round({'electricity': 1, 'pred_elec': 1, 'water': 1, 'pred_water': 1}))
