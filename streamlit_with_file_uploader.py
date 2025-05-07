import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
import io

# File uploader instead of fixed path
uploaded_file = st.sidebar.file_uploader("Upload your cleaned dataset (CSV)", type="csv")

# Load data function
@st.cache_data  # Updated from st.cache (which is deprecated)
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        return None

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Welcome", "Data Overview", "Exploratory Data Analysis", "Model Building"])

# Main content
if uploaded_file is None:
    st.title("Welcome to the Air Quality Analysis App")
    st.write("Please upload your CSV file using the sidebar to get started.")
else:
    # Load the data
    df = load_data(uploaded_file)
    
    # Main page content
    if selection == "Welcome":
        st.title("Welcome to the Air Quality Analysis App")
        st.write("Your data has been successfully loaded! Use the sidebar to navigate through the app.")
        st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    
    elif selection == "Data Overview":
        st.title("Data Overview")
        st.write("Here's a glimpse of the dataset:")
        st.dataframe(df.head())
        
        # Display info in a more Streamlit-friendly way
        st.write("Dataset Information:")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.write("Statistical Summary:")
        st.write(df.describe())
    
    elif selection == "Exploratory Data Analysis":
        st.title("Exploratory Data Analysis")
        
        st.subheader("Shape & Size")
        st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
        
        st.subheader("Univariate Analysis")
        column = st.selectbox("Select a column for univariate analysis", df.columns)
        if pd.api.types.is_numeric_dtype(df[column]):
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.write(f"Column '{column}' is not numeric. Showing value counts instead:")
            st.write(df[column].value_counts())
        
        st.subheader("Bivariate Analysis")
        col1 = st.selectbox("Select X-axis variable", df.columns, key='biv_x')
        col2 = st.selectbox("Select Y-axis variable", df.columns, key='biv_y')
        
        # Check if both columns are numeric
        if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
            st.pyplot(fig)
        else:
            st.write("Both columns must be numeric for scatter plot.")
        
        # Only show scatter matrix for numeric columns
        st.subheader("Multivariable Analysis")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) > 1:
            # Limit to first 8 numeric columns to avoid performance issues
            display_cols = numeric_cols[:8] if len(numeric_cols) > 8 else numeric_cols
            fig = px.scatter_matrix(df[display_cols])
            st.plotly_chart(fig)
        else:
            st.write("Not enough numeric columns for scatter matrix analysis.")
        
        # Time series requires datetime column
        st.subheader("Time Series Analysis")
        # Check if any column might be a datetime
        datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if datetime_cols:
            default_time_col = datetime_cols[0]
        else:
            default_time_col = df.columns[0]
            
        time_column = st.selectbox("Select time column", df.columns, index=df.columns.get_loc(default_time_col))
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df[time_column]):
            try:
                df[time_column] = pd.to_datetime(df[time_column])
                st.write(f"Converted '{time_column}' to datetime format.")
            except:
                st.warning(f"Could not convert '{time_column}' to datetime. Using as-is.")
        
        value_column = st.selectbox("Select value column", 
                                   df.select_dtypes(include=['number']).columns)
        
        if len(df) > 0:
            fig = px.line(df.sort_values(by=time_column), x=time_column, y=value_column)
            st.plotly_chart(fig)
    
    elif selection == "Model Building":
        st.title("Model Building")
        
        # Get numeric columns for modeling
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("Need at least two numeric columns for modeling (one for target, one for features).")
        else:
            st.subheader("Splitting Data")
            target = st.selectbox("Select target variable", numeric_cols)
            available_features = [col for col in numeric_cols if col != target]
            
            # Default to selecting all available features
            features = st.multiselect("Select feature variables", available_features, default=available_features[:3])
            
            if not features:
                st.warning("Please select at least one feature variable.")
            else:
                X = df[features]
                y = df[target]
                
                # Handle missing values for modeling
                if X.isna().any().any() or y.isna().any():
                    st.warning("Data contains missing values. Removing rows with missing values for modeling.")
                    valid_indices = ~(X.isna().any(axis=1) | y.isna())
                    X = X[valid_indices]
                    y = y[valid_indices]
                
                test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
                random_state = st.slider("Random state", 0, 100, 42)
                
                if len(X) > 1 and len(set(y)) > 1:  # Ensure we have enough data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    st.write(f"Training set: {X_train.shape[0]} samples")
                    st.write(f"Testing set: {X_test.shape[0]} samples")
                    
                    # Target distribution plot
                    st.subheader(f"{target} Distribution Plot (Train vs Test)")
                    fig, ax = plt.subplots()
                    sns.kdeplot(y_train, label='Train', ax=ax)
                    sns.kdeplot(y_test, label='Test', ax=ax)
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Linear Regression
                    st.subheader("Linear Regression")
                    lr = LinearRegression()
                    with st.spinner("Training Linear Regression model..."):
                        lr.fit(X_train, y_train)
                        y_pred_lr = lr.predict(X_test)
                        mse_lr = mean_squared_error(y_test, y_pred_lr)
                        rmse_lr = mse_lr ** 0.5
                    
                    st.write(f"Linear Regression Mean Squared Error: {mse_lr:.4f}")
                    st.write(f"Linear Regression Root Mean Squared Error: {rmse_lr:.4f}")
                    
                    # Feature importance
                    coef_df = pd.DataFrame({
                        'Feature': features,
                        'Coefficient': lr.coef_
                    }).sort_values('Coefficient', ascending=False)
                    
                    st.write("Feature Importance:")
                    st.dataframe(coef_df)
                    
                    # K-Nearest Neighbors Regression
                    st.subheader("K-Nearest Neighbors Regression")
                    k = st.slider("Select number of neighbors (k)", 1, 20, 5)
                    knn = KNeighborsRegressor(n_neighbors=k)
                    
                    with st.spinner("Training KNN model..."):
                        knn.fit(X_train, y_train)
                        y_pred_knn = knn.predict(X_test)
                        mse_knn = mean_squared_error(y_test, y_pred_knn)
                        rmse_knn = mse_knn ** 0.5
                    
                    st.write(f"KNN Regression Mean Squared Error: {mse_knn:.4f}")
                    st.write(f"KNN Regression Root Mean Squared Error: {rmse_knn:.4f}")
                    
                    # Model Comparison
                    st.subheader("Model Comparison")
                    model_comparison = pd.DataFrame({
                        'Model': ['Linear Regression', 'KNN Regression'],
                        'MSE': [mse_lr, mse_knn],
                        'RMSE': [rmse_lr, rmse_knn]
                    })
                    st.dataframe(model_comparison)
                    
                    # Predictions vs Actual
                    st.subheader("Predictions vs Actual")
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Linear Regression
                    ax[0].scatter(y_test, y_pred_lr)
                    ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
                    ax[0].set_xlabel('Actual')
                    ax[0].set_ylabel('Predicted')
                    ax[0].set_title('Linear Regression')
                    
                    # KNN
                    ax[1].scatter(y_test, y_pred_knn)
                    ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
                    ax[1].set_xlabel('Actual')
                    ax[1].set_ylabel('Predicted')
                    ax[1].set_title('KNN Regression')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.error("Not enough valid data for modeling after handling missing values.")
