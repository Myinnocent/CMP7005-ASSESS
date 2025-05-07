import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px

# File uploader
st.title("Air Quality Analysis App")
uploaded_file = st.file_uploader("Upload your cleaned_dataset.csv file", type=["csv"])

# Load dataset with caching
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Check if file is uploaded
if uploaded_file is not None:
    df = load_data(uploaded_file)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Welcome", "Data Overview", "Exploratory Data Analysis", "Model Building"])

    # Main page content
    if selection == "Welcome":
        st.title("Welcome to the Air Quality Analysis App")
        st.write("Use the sidebar to navigate through the app.")

    elif selection == "Data Overview":
        st.title("Data Overview")
        st.write("Here's a glimpse of the dataset:")
        st.dataframe(df.head())
        st.write("Dataset Information:")
        buffer = df.info(buf=None)
        st.text(buffer)
        st.write("Statistical Summary:")
        st.write(df.describe())

    elif selection == "Exploratory Data Analysis":
        st.title("Exploratory Data Analysis")

        st.subheader("Shape & Size")
        st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

        st.subheader("Univariate Analysis")
        column = st.selectbox("Select a column for univariate analysis", df.columns)
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Bivariate Analysis")
        col1 = st.selectbox("Select X-axis variable", df.columns, key='biv_x')
        col2 = st.selectbox("Select Y-axis variable", df.columns, key='biv_y')
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
        st.pyplot(fig)

        st.subheader("Multivariable Analysis")
        fig = px.scatter_matrix(df)
        st.plotly_chart(fig)

        st.subheader("Time Series Analysis")
        time_column = st.selectbox("Select time column", df.columns)
        value_column = st.selectbox("Select value column", df.columns)
        fig = px.line(df, x=time_column, y=value_column)
        st.plotly_chart(fig)

    elif selection == "Model Building":
        st.title("Model Building")

        st.subheader("Splitting Data")
        target = st.selectbox("Select target variable", df.columns)
        features = st.multiselect("Select feature variables", df.columns.drop(target))
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"Training set: {X_train.shape[0]} samples")
        st.write(f"Testing set: {X_test.shape[0]} samples")

        st.subheader("PM2.5 Distribution Plot (Train vs Test)")
        if 'PM2.5' in df.columns:
            fig, ax = plt.subplots()
            sns.kdeplot(y_train, label='Train', ax=ax)
            sns.kdeplot(y_test, label='Test', ax=ax)
            ax.legend()
            st.pyplot(fig)
        else:
            st.write("PM2.5 column not found in the dataset.")

        st.subheader("Linear Regression")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        st.write(f"Linear Regression Mean Squared Error: {mse_lr}")

        st.subheader("K-Nearest Neighbors Regression")
        k = st.slider("Select number of neighbors (k)", 1, 20, 5)
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        mse_knn = mean_squared_error(y_test, y_pred_knn)
        st.write(f"KNN Regression Mean Squared Error: {mse_knn}")
else:
    st.warning("Please upload the dataset to continue.")
