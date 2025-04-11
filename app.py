import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Housing Price Prediction with Linear Regression')

@st.cache_data
def load_data():
    return pd.read_csv('housing.csv')

def main():
    try:
        df = load_data()
        st.write("### Data Overview")
        st.write(df.head())

        X = df['total_rooms'].values.reshape(-1, 1)
        y = df['median_house_value'].values

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        st.write("### Visualization of Linear Regression")
        if st.button("Show Graph"):
            fig, _ = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=X.flatten(), y=y, alpha=0.5)
            plt.plot(X, y_pred, color='red', linewidth=2)
            plt.xlabel('Total Rooms')
            plt.ylabel('Median House Value')
            plt.title('Linear Regression: House Value vs Total Rooms')
            st.pyplot(fig)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        st.write("### Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Absolute Error", f"${mae:,.2f}")
        with col2:
            st.metric("Root Mean Squared Error", f"${rmse:,.2f}")
        with col3:
            st.metric("R² Score", f"{r2:.3f}")

        st.write("### Model Interpretation")
        if r2 < 0.3:
            st.write("Based on the R² score, total_rooms appears to be a weak predictor of median_house_value.")
        elif r2 < 0.7:
            st.write("Based on the R² score, total_rooms appears to be a moderate predictor of median_house_value.")
        else:
            st.write("Based on the R² score, total_rooms appears to be a strong predictor of median_house_value.")

        st.write("### Regression Equation")
        st.write(f"median_house_value = {model.intercept_:.2f} + {model.coef_[0]:.2f} × total_rooms")

        st.write("### Predict Median House Value")
        total_rooms_input = st.slider(
            "Select the number of total rooms:",
            min_value=0,
            max_value=int(df['total_rooms'].max()),
            value=1000,
            step=1
        )
        if total_rooms_input > 0:
            predicted_value = model.predict([[total_rooms_input]])[0]
            st.write(f"Predicted Median House Value: ${predicted_value:,.2f}")

        st.write("### Predict Using Regression Equation")
        regression_slider = st.slider(
            "Select the number of total rooms for regression equation:",
            min_value=0,
            max_value=int(df['total_rooms'].max()),
            value=1000,
            step=1
        )
        if regression_slider > 0:
            regression_predicted_value = model.intercept_ + model.coef_[0] * regression_slider
            st.write(f"Predicted Median House Value (Regression Equation): ${regression_predicted_value:,.2f}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please make sure the 'housing.csv' file is in the same directory as this script.")

if __name__ == "__main__":
    main()