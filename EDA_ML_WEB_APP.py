import base64
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from scipy.stats import pearsonr,chi2_contingency
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score,GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error,r2_score, mean_absolute_error,accuracy_score, precision_score, recall_score, f1_score,ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
import joblib
import os
import pickle
from sklearn.metrics import roc_auc_score,roc_curve

# Add custom CSS to the Streamlit app
st.markdown(
    """
    <style>
    /* Define the font-face with Times New Roman */
    @font-face {
        font-family: 'Times New Roman';
        src: url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
    }

    /* Apply the font-family to the Streamlit body */
    * {
        font-family: 'Times New Roman', serif;
    }
    </style>
    """,
    unsafe_allow_html=True)

# Sidebar Section: Contact Information
st.sidebar.subheader('Contact Information')

# Add your contact details
st.sidebar.markdown('**Name:** Amro Ewes')
st.sidebar.markdown('**Email:** amroabousree@gmail.com')
st.sidebar.markdown('**LinkedIn:** [linkedin.com/in/amro-ewes](https://www.linkedin.com/in/amro-ewes-663723162/)')
st.sidebar.markdown('**Phone:** +965 6047-8385')
st.sidebar.markdown('**GitHub:** [github.com/AmroEwes](https://github.com/AmroEwes)')

#Detecting Outliers
def extract_outliers_from_boxplot(array):
    ## Get IQR
    array = array[np.logical_not(np.isnan(array))]
    iqr_q1 = np.quantile(array, 0.25)
    iqr_q3 = np.quantile(array, 0.75)
    med = np.median(array)

    # finding the iqr region
    iqr = iqr_q3-iqr_q1

    # finding upper and lower whiskers
    upper_bound = iqr_q3+(1.5*iqr)
    lower_bound = iqr_q1-(1.5*iqr)

    outliers = array[(array <= lower_bound) | (array >= upper_bound)]
    return outliers.index ,outliers.sort_values()

def download_data_as_csv(data):
    csv_data = data.to_csv(index=False)
    b64_data = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64_data}" download="data.csv">Download CSV</a>'
    return href


# Function for chi-square feature selection
def chi_square_feature_selection(data, features, target, k):
    X = data[features]
    y = data[target]
    selector = SelectKBest(score_func=chi2 , k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = np.array(features)[selector.get_support()].tolist()

    chi2_scores = []
    p_values = []
    for feature in features:
        contingency_table = pd.crosstab(data[feature], data[target])
        chi2_, p_value, _, _ = chi2_contingency(contingency_table)
        chi2_scores.append(chi2_)
        p_values.append(p_value)
    results = pd.DataFrame({'Feature': features, 'Chi-square': chi2_scores, 'P-value': p_values})
    results = results.sort_values(by='Chi-square', ascending=False).reset_index(drop=True)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Feature', y='Chi-square', data=results)
    plt.title('Chi-square Scores for Features')
    plt.xticks(rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Chi-square Score')
    st.pyplot(fig)

    return selected_features, pd.DataFrame(X_new, columns=selected_features)


# Function for ANOVA feature selection
def anova_feature_selection(data, features, target, k):
    X = data[features]
    y = data[target]
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = np.array(features)[selector.get_support()].tolist()

    f_scores, p_values = f_classif(X, y)
    results = pd.DataFrame({'Feature': features, 'F-score': f_scores, 'P-value': p_values})
    results = results.sort_values(by='F-score', ascending=False).reset_index(drop=True)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Feature', y='F-score', data=results)
    plt.title('F-scores for Features (ANOVA)')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('F-score')
    st.pyplot(fig)

    return selected_features, pd.DataFrame(X_new, columns=selected_features)


# Function for correlation-based feature selection
def correlation_feature_selection(data, features, target, threshold):
    selected_features = []
    correlations = []
    for feature in features:
        corr, p_value = pearsonr(data[feature], data[target])
        if abs(corr) >= threshold:
            selected_features.append(feature)
        correlations.append(corr)

    results = pd.DataFrame({'Feature': features, 'Correlation': correlations})
    results = results.sort_values(by='Correlation', ascending=False).reset_index(drop=True)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Feature', y='Correlation', data=results)
    plt.title('Correlations with Target Variable')
    plt.xticks(rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Correlation')
    st.pyplot(fig)

    return selected_features, data[selected_features]

# Function for PCA-based feature selection
def pca_feature_selection(data, features, n_components):
    
    X = data[features]
    pca = PCA(n_components=n_components)
    X_new = pca.fit_transform(X)
    selected_features = [f"PC{i+1}" for i in range(n_components)]

    # Explained variance ratio
    explained_var = pca.explained_variance_ratio_

    # Visualization
    fig,ax = plt.subplots(figsize=(10, 6))
    plt.bar(range(1, n_components + 1), explained_var)
    plt.title('Explained Variance Ratio (PCA)')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    st.pyplot(fig)

    return selected_features, pd.DataFrame(X_new, columns=selected_features)

def split_data(data, features, target, test_size, random_state):
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the y values for both training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate the coefficient of determination (R-squared) for training and test data
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Calculate the mean absolute error for training and test data
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Calculate the mean squared error for training and test data
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # Calculate the model accuracy
    accuracy_train = model.score(X_train, y_train)
    accuracy_test = model.score(X_test, y_test)
    
    result = pd.DataFrame({
        'Metric': ['R-squared', 'Mean Absolute Error', 'Mean Squared Error'],
        'Train': [r2_train, mae_train, mse_train],
        'Test': [r2_test, mae_test, mse_test]
    })
    st.write("Linear Regression Results:")
    st.dataframe(result)

    # Plot the actual vs. predicted values for test data
    fig,ax = plt.subplots(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, color='blue', label='Actual vs. Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    st.pyplot(fig)

def lasso_regression(X_train, X_test, y_train, y_test):
    model = Lasso()
    model.fit(X_train, y_train)

    # Predict the y values for both training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate the coefficient of determination (R-squared) for training and test data
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Calculate the mean absolute error for training and test data
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Calculate the mean squared error for training and test data
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # Calculate the model accuracy
    accuracy_train = model.score(X_train, y_train)
    accuracy_test = model.score(X_test, y_test)
    
    result = pd.DataFrame({
        'Metric': ['R-squared', 'Mean Absolute Error', 'Mean Squared Error'],
        'Train': [r2_train, mae_train, mse_train],
        'Test': [r2_test, mae_test, mse_test]
    })
    st.write("Lasso Results:")
    st.dataframe(result)

    # Plot the actual vs. predicted values for test data
    fig,ax = plt.subplots(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, color='blue', label='Actual vs. Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    st.pyplot(fig)

def ridge_regression(X_train, X_test, y_train, y_test):
    model = Ridge()
    model.fit(X_train, y_train)

    # Predict the y values for both training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate the coefficient of determination (R-squared) for training and test data
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Calculate the mean absolute error for training and test data
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Calculate the mean squared error for training and test data
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # Calculate the model accuracy
    accuracy_train = model.score(X_train, y_train)
    accuracy_test = model.score(X_test, y_test)
    
    result = pd.DataFrame({
        'Metric': ['R-squared', 'Mean Absolute Error', 'Mean Squared Error'],
        'Train': [r2_train, mae_train, mse_train],
        'Test': [r2_test, mae_test, mse_test]
    })
    st.write("Ridge Results:")
    st.dataframe(result)

    # Plot the actual vs. predicted values for test data
    fig,ax = plt.subplots(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, color='blue', label='Actual vs. Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    st.pyplot(fig)

def Random_Forest_regression(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predict the y values for both training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate the coefficient of determination (R-squared) for training and test data
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Calculate the mean absolute error for training and test data
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Calculate the mean squared error for training and test data
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # Calculate the model accuracy
    accuracy_train = model.score(X_train, y_train)
    accuracy_test = model.score(X_test, y_test)
    
    result = pd.DataFrame({
        'Metric': ['R-squared', 'Mean Absolute Error', 'Mean Squared Error'],
        'Train': [r2_train, mae_train, mse_train],
        'Test': [r2_test, mae_test, mse_test]
    })
    st.write("RandomForestRegressor Results:")
    st.dataframe(result)

    # Plot the actual vs. predicted values for test data
    fig,ax = plt.subplots(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, color='blue', label='Actual vs. Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    st.pyplot(fig)

def XGBOOST_regression(X_train, X_test, y_train, y_test):
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Predict the y values for both training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate the coefficient of determination (R-squared) for training and test data
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Calculate the mean absolute error for training and test data
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Calculate the mean squared error for training and test data
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # Calculate the model accuracy
    accuracy_train = model.score(X_train, y_train)
    accuracy_test = model.score(X_test, y_test)
    
    result = pd.DataFrame({
        'Metric': ['R-squared', 'Mean Absolute Error', 'Mean Squared Error'],
        'Train': [r2_train, mae_train, mse_train],
        'Test': [r2_test, mae_test, mse_test]
    })
    st.write("XGBRegressor Results:")
    st.dataframe(result)

    # Plot the actual vs. predicted values for test data
    fig,ax = plt.subplots(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, color='blue', label='Actual vs. Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    st.pyplot(fig)

def SVM_regression(X_train, X_test, y_train, y_test):
    model = SVR()
    model.fit(X_train, y_train)

    # Predict the y values for both training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate the coefficient of determination (R-squared) for training and test data
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Calculate the mean absolute error for training and test data
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Calculate the mean squared error for training and test data
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # Calculate the model accuracy
    accuracy_train = model.score(X_train, y_train)
    accuracy_test = model.score(X_test, y_test)
    
    result = pd.DataFrame({
        'Metric': ['R-squared', 'Mean Absolute Error', 'Mean Squared Error'],
        'Train': [r2_train, mae_train, mse_train],
        'Test': [r2_test, mae_test, mse_test]
    })
    st.write("SVR Results:")
    st.dataframe(result)

    # Plot the actual vs. predicted values for test data
    fig,ax = plt.subplots(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, color='blue', label='Actual vs. Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    st.pyplot(fig)


def logistic_regression(x_train, x_test, y_train, y_test):
    # Fit the Random Forest Classifier model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    # Predict the classes for both training and test data
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # Calculate the accuracy, precision, recall, and F1-score for training and test data
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_train = precision_score(y_train, y_train_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_train = recall_score(y_train, y_train_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)
    
    # Calculate the confusion matrix for test data
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Create a DataFrame to store the statistical findings
    result = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Train': [accuracy_train, precision_train, recall_train, f1_train],
        'Test': [accuracy_test, precision_test, recall_test, f1_test]})
    
    # Get the classification report as a string
    report = classification_report(y_test, y_test_pred, output_dict=True)
    
    # Convert the classification report to a DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    st.subheader("Classification Report Results: -")
    st.dataframe(report_df)

    st.write("LogisticRegression Results:")
    st.dataframe(result)
    # Display the confusion matrix using Streamlit
    st.write("Confusion Matrix:")
    st.write(cm)
    fig,ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, cmap='Blues', annot=True,fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    st.pyplot(fig)

    fpr, tpr,_ = roc_curve(y_test,  y_test_pred)
    auc = roc_auc_score(y_test, y_test_pred)

     # Create a DataFrame for ROC curve data
    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'AUC': auc})
    
    st.write("ROC Curve:")
    st.dataframe(roc_data)

    st.subheader("ROC Visualization")
    fig,ax = plt.subplots(figsize=(10, 6))
    plt.plot(fpr, tpr,label="AUC="+str(auc),color = 'green')
    plt.legend(loc=4)
    plt.title('ROC Score')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    st.pyplot(fig)

def random_forest_classifier(x_train, x_test, y_train, y_test):
    # Fit the Random Forest Classifier model
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    
    # Predict the classes for both training and test data
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # Calculate the accuracy, precision, recall, and F1-score for training and test data
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_train = precision_score(y_train, y_train_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_train = recall_score(y_train, y_train_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)
    
    # Calculate the confusion matrix for test data
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Create a DataFrame to store the statistical findings
    result = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Train': [accuracy_train, precision_train, recall_train, f1_train],
        'Test': [accuracy_test, precision_test, recall_test, f1_test]})
    
    # Get the classification report as a string
    report = classification_report(y_test, y_test_pred, output_dict=True)
    
    # Convert the classification report to a DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    st.subheader("Classification Report Results: -")
    st.dataframe(report_df)

    st.write("RandomForestClassifier Results:")
    st.dataframe(result)
    # Display the confusion matrix using Streamlit
    st.write("Confusion Matrix:")
    st.write(cm)
    fig,ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, cmap='Blues', annot=True,fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    st.pyplot(fig)

    fpr, tpr,_ = roc_curve(y_test,  y_test_pred)
    auc = roc_auc_score(y_test, y_test_pred)

     # Create a DataFrame for ROC curve data
    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'AUC': auc})
    
    st.write("ROC Curve:")
    st.dataframe(roc_data)

    st.subheader("ROC Visualization")
    fig,ax = plt.subplots(figsize=(10, 6))
    plt.plot(fpr, tpr,label="AUC="+str(auc),color = 'green')
    plt.legend(loc=4)
    plt.title('ROC Score')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    st.pyplot(fig)

def SVM_classifier(x_train, x_test, y_train, y_test):
    # Fit the Random Forest Classifier model
    model = SVC()
    model.fit(x_train, y_train)
    
    # Predict the classes for both training and test data
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # Calculate the accuracy, precision, recall, and F1-score for training and test data
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_train = precision_score(y_train, y_train_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_train = recall_score(y_train, y_train_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)
    
    # Calculate the confusion matrix for test data
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Create a DataFrame to store the statistical findings
    result = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Train': [accuracy_train, precision_train, recall_train, f1_train],
        'Test': [accuracy_test, precision_test, recall_test, f1_test]})
    
    # Get the classification report as a string
    report = classification_report(y_test, y_test_pred, output_dict=True)
    
    # Convert the classification report to a DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    st.subheader("Classification Report Results: -")
    st.dataframe(report_df)

    st.write("SVR Results:")
    st.dataframe(result)
    # Display the confusion matrix using Streamlit
    st.write("Confusion Matrix:")
    st.write(cm)
    fig,ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, cmap='Blues', annot=True,fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    st.pyplot(fig)

    fpr, tpr,_ = roc_curve(y_test,  y_test_pred)
    auc = roc_auc_score(y_test, y_test_pred)

     # Create a DataFrame for ROC curve data
    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'AUC': auc})
    
    st.write("ROC Curve:")
    st.dataframe(roc_data)

    st.subheader("ROC Visualization")
    fig,ax = plt.subplots(figsize=(10, 6))
    plt.plot(fpr, tpr,label="AUC="+str(auc),color = 'green')
    plt.legend(loc=4)
    plt.title('ROC Score')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    st.pyplot(fig)

def XGBOOST_classifier(x_train, x_test, y_train, y_test):
    # Fit the Random Forest Classifier model
    model = XGBClassifier()
    model.fit(x_train, y_train)
    
    # Predict the classes for both training and test data
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # Calculate the accuracy, precision, recall, and F1-score for training and test data
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_train = precision_score(y_train, y_train_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_train = recall_score(y_train, y_train_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)
    
    # Calculate the confusion matrix for test data
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Create a DataFrame to store the statistical findings
    result = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Train': [accuracy_train, precision_train, recall_train, f1_train],
        'Test': [accuracy_test, precision_test, recall_test, f1_test]})
    
    # Get the classification report as a string
    report = classification_report(y_test, y_test_pred, output_dict=True)
    
    # Convert the classification report to a DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    st.subheader("Classification Report Results: -")
    st.dataframe(report_df)


    st.write("XGBClassifier Results:")
    st.dataframe(result)
    # Display the confusion matrix using Streamlit
    st.write("Confusion Matrix:")
    st.write(cm)
    fig,ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, cmap='Blues', annot=True,fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    st.pyplot(fig)

    fpr, tpr,_ = roc_curve(y_test,  y_test_pred)
    auc = roc_auc_score(y_test, y_test_pred)

     # Create a DataFrame for ROC curve data
    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'AUC': auc})
    
    st.write("ROC Curve:")
    st.dataframe(roc_data)

    st.subheader("ROC Visualization")
    fig,ax = plt.subplots(figsize=(10, 6))
    plt.plot(fpr, tpr,label="AUC="+str(auc),color = 'green')
    plt.legend(loc=4)
    plt.title('ROC Score')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    st.pyplot(fig)


def Decision_tree(x_train, x_test, y_train, y_test):
    # Fit the Random Forest Classifier model
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    
    # Predict the classes for both training and test data
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # Calculate the accuracy, precision, recall, and F1-score for training and test data
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_train = precision_score(y_train, y_train_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_train = recall_score(y_train, y_train_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)
    
    # Calculate the confusion matrix for test data
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Create a DataFrame to store the statistical findings
    result = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Train': [accuracy_train, precision_train, recall_train, f1_train],
        'Test': [accuracy_test, precision_test, recall_test, f1_test]})
    
    # Get the classification report as a string
    report = classification_report(y_test, y_test_pred, output_dict=True)
    
    # Convert the classification report to a DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    st.subheader("Classification Report Results: -")
    st.dataframe(report_df)

    st.subheader("DecisionTreeClassifier Results:")
    st.dataframe(result)
    # Display the confusion matrix using Streamlit
    st.write("Confusion Matrix:")
    st.write(cm)
    fig,ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, cmap='Blues', annot=True,fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    st.pyplot(fig)

    fpr, tpr,_ = roc_curve(y_test,  y_test_pred)
    auc = roc_auc_score(y_test, y_test_pred)

     # Create a DataFrame for ROC curve data
    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'AUC': auc})
    
    st.write("ROC Curve:")
    st.dataframe(roc_data)

    st.subheader("ROC Visualization")
    fig,ax = plt.subplots(figsize=(10, 6))
    plt.plot(fpr, tpr,label="AUC="+str(auc),color = 'green')
    plt.legend(loc=4)
    plt.title('ROC Score')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    st.pyplot(fig)

# Define a function for hyperparameter tuning and evaluation
def hyperparameter_tuning(model, param_grid, search_type, X_train, X_test, y_train, y_test):
    if search_type == "Grid Search":
    # Perform grid search cross-validation
        search = GridSearchCV(model, param_grid, cv=5)
    elif search_type == "Random Search":
        # Perform random search cross-validation
        search = RandomizedSearchCV(model, param_grid, cv=5)

    search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = search.best_estimator_
    best_params = search.best_params_

    with open('tuned_model.pickle', 'wb') as f:
        pickle.dump(best_model, f)

    # Evaluate the model on the test set
    accuracy = best_model.score(X_test, y_test)

    # Create a data frame to store the results
    results_df = pd.DataFrame({'Model': [selected_algorithm],'Search Type': [search_type],
    'Best Parameters': [best_params],'Accuracy': [accuracy]})

    st.write("Hyperparameter Tuning results:")
    st.dataframe(results_df)

    if st.checkbox("Download your tuned model"):
        with open('tuned_model.pickle', 'rb') as f:
            model_bytes = f.read()
        st.download_button(
            label="Download Tuned Model",
            data=model_bytes,
            file_name='tuned_model.pickle',
            mime='application/octet-stream'
        )


options = st.sidebar.radio("Select an option", ["User Guide Page","Exploratory Data Analysis","Preprocessing & Data Manipulation","Feature Selection","ML Models"])


if options == "User Guide Page":
    st.title("Welcome to EDA & ML web app")
    user_guide = st.selectbox("Select the required page",["Please select the required page!","Exploratory Data Analysis","Preprocessing & Data Manipulation","Feature Selection","ML Models"])
    if user_guide == "Please select the required page!":
        st.info("No page selected")
    elif user_guide ==  "Exploratory Data Analysis":
        st.write("""
        ### User Guide for Exploratory Data Analysis Page:

        The "Exploratory Data Analysis" page allows you to gain insights into your dataset. Follow the steps below to get started:

        1. Upload Data: Click on the "Upload a CSV file" button to select and upload your dataset in CSV format.

        2. Random Sample from the Data: Enable the "Random Sample from the Data" checkbox to view a random sample of five rows from the selected columns of your dataset.

        3. Data Shape: Enable the "Data Shape" checkbox to display the shape of the selected columns. It shows the number of rows and columns in your dataset.

        4. Missing Values: Enable the "Missing Values" checkbox to view the number of missing values in each selected column.

        5. Unique Values: Enable the "Unique values" checkbox to see the count of unique values in each selected column.

        6. Statistical Summary: Enable the "Statistical Summary" checkbox to obtain a statistical summary of the selected columns. It includes count, mean, standard deviation, minimum, 25th percentile (Q1), median, 75th percentile (Q3), and maximum values.

        7. Data Types: Enable the "Data Types" checkbox to display the data types of the selected columns.

        8. Detecting Outliers: Enable the "Detecting Outliers" checkbox to identify outliers in the numerical columns of your dataset. Select a feature from the dropdown menu to view the outliers detected using the box plot method.

        9. Data Visualization: Enable the "Data Visualization" checkbox to explore visualizations of your data. Choose a feature and a target variable from the dropdown menus. Then select a plot type such as scatter plot, histogram, box plot, bar chart, or line plot to visualize the relationship between the selected features.

        """)
    elif user_guide == "Preprocessing & Data Manipulation":

        st.write("""
        ### User Guide for Preprocessing & Data Manipulation Page:

        The "Data Preprocessing" page allows you to preprocess and manipulate your dataset. Follow the steps below to get started:

        1. Upload Data: Click on the "Upload a CSV file" button to select and upload your dataset in CSV format.

        2. Handling Missing Values: Select the columns that contain missing values and choose a fill method such as dropping null values, filling with mean, median, mode, backward fill, forward fill, maximum value, or minimum value.

        3. Labeling Categorical Columns: Identify the categorical columns and label them using methods like Label Encoder or OneHot Encoding.

        4. Remove Duplicates: Select the columns based on which you want to remove duplicate rows from your dataset.

        5. Delete Columns: Choose the columns that you want to delete from your dataset.

        6. Rename Columns: Select columns and provide new names to rename them.

        7. Download Processed Data: Download the processed data as a CSV file.

        """)

    elif user_guide == "Feature Selection":

        st.write("""
        ### User Guide for Feature Selection Page:

        The "Feature Selection" page allows you to select the most relevant features from your dataset. Follow the steps below to get started:

        1. Upload Data: Click on the "Upload a CSV file" button to select and upload your dataset in CSV format.

        2. Select Features and Target: Choose the features and the target variable from your dataset.

        3. Apply Feature Selection: Select the desired feature selection method and specify the parameters if applicable.

        4. Display Selected Features and New Data: View the selected features and the transformed dataset.

        5. Download New Data: Download the transformed dataset as a CSV file.

        **Note**: Ensure that you have uploaded a CSV file and selected the required options before proceeding with feature selection. The feature selection methods available are:
        - Chi-square: Select the number of features using the chi-square statistical test.
        - ANOVA: Select the number of features using the ANOVA F-test.
        - Correlation: Select features based on their correlation with the target variable.
        - PCA: Select features using Principal Component Analysis.

        After applying feature selection, the selected features will be displayed, and you can download the transformed dataset.

        """)
    elif user_guide == "ML Models":

        st.write("""
        ### User Guide for ML Models Page:

        The "ML Models" page allows you to train and evaluate machine learning models on your dataset. Follow the steps below to get started:

        1. Upload Data: Click on the "Upload a CSV file" button to select and upload your dataset in CSV format.

        2. Select Features and Target: Choose the features and the target variable from your dataset.

        3. Standardization (Optional): Select the columns to standardize using either the Standard Scaler or Min-Max Scaler method.

        4. Select Problem Type: Choose whether your problem is a regression or classification problem.

        5. Choose ML Algorithm: Select the desired machine learning algorithm for regression or classification.

        6. Hyperparameter Tuning (Optional): Apply hyperparameter tuning to optimize the model performance.

        **Note**: Ensure that you have uploaded a CSV file, selected the required options, and applied any desired standardization before proceeding with model training. The machine learning algorithms available are:
        - Regression: Choose from Linear Regression, Lasso Regression, Ridge Regression, Random Forest Regressor, XGBoost Regressor, or SVM Regressor.
        - Classification: Choose from Logistic Regression, SVM Classifier, Random Forest Classifier, XGBoost Classifier, or Decision Tree.

        After training the model, the evaluation metrics and predicted values (for regression) or the confusion matrix and classification report (for classification) will be displayed.

        If you enable hyperparameter tuning, the best parameters and the corresponding evaluation metrics will be shown.

        """)

elif options == "Exploratory Data Analysis":
    # Upload data
    st.header("Upload Data")
    # Ask the user to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        # Ask the user to select the required columns
        df = pd.read_csv(uploaded_file)
        selected_columns = st.multiselect("Select columns", df.columns.tolist())

        # Random Data Sample
        data_sample = st.checkbox("Random Sample from the Data:")
        if data_sample:
            st.dataframe(df[selected_columns].sample(5))

        # Data shape
        data_shape = st.checkbox("Data Shape:")
        if data_shape:
            st.write(df[selected_columns].shape)

        # Missing Values
        missing_values = st.checkbox("Missing Values:")
        if missing_values:
            st.write(df[selected_columns].isnull().sum())

        # Unique values
        total_unique_values = st.checkbox("Unique values:")
        if total_unique_values:
            st.write(df[selected_columns].nunique())

        # Statistical Summary
        statistal_summary = st.checkbox("Statistical Summary:")
        if statistal_summary:
            st.write(df[selected_columns].describe())

        # Data Types
        data_types = st.checkbox("Data Types:")
        if data_types:
            st.write(df[selected_columns].dtypes)

        # Dedecting Outliers
        not_categorical = [cols for cols in selected_columns if df[cols].dtype != "object"]
        detect_outliers = st.checkbox("Dedecting Outliers:")
        if detect_outliers:
            not_categorical_selected = st.selectbox("Select Feature",["No Feature Selected"] + not_categorical)
            if not_categorical_selected == "No Feature Selected":
                st.warning("Please Select the requiered feature")
            else:
                st.dataframe(extract_outliers_from_boxplot(df[not_categorical_selected]))

        # Data Visualization
        if st.checkbox("Data Visualization"):
            feature = st.selectbox("Select feature", selected_columns)
            target = st.selectbox("Select target", selected_columns)
            plot_type = st.selectbox("Select plot type", ["No Plot Selected","Scatter plot", "Histogram", "Box Plot", "Bar Chart", "Line plot"])

            if plot_type == "No Plot Selected":
                st.warning("Please Select the plot type")

            elif plot_type == "Scatter plot":
                fig = px.scatter(df, x=feature, y=target)
                fig.update_layout(
                    title=f"Scatter plot of {feature} vs {target}",
                    xaxis_title=feature,
                    yaxis_title=target,
                    showlegend=True
                )
                st.plotly_chart(fig)
            elif plot_type == "Histogram":
                fig = px.histogram(df, x=feature)
                fig.update_layout(
                    title=f"Histogram of {feature}",
                    xaxis_title=feature,
                    yaxis_title="Count",
                    showlegend=False
                )
                st.plotly_chart(fig)
            elif plot_type == "Box Plot":
                fig = px.box(df, x=target, y=feature)
                fig.update_layout(
                    title=f"Box Plot of {feature} by {target}",
                    xaxis_title=target,
                    yaxis_title=feature,
                    showlegend=False
                )
                st.plotly_chart(fig)
            elif plot_type == "Bar Chart":
                fig = px.bar(df, x=feature, y=target)
                fig.update_layout(
                    title=f"Bar Chart of {feature} by {target}",
                    xaxis_title=feature,
                    yaxis_title=target,
                    showlegend=False
                )
                st.plotly_chart(fig)
            elif plot_type == "Line plot":
                fig = px.line(df, x=feature, y=target)
                fig.update_layout(
                    title=f"Line Plot of {feature} by {target}",
                    xaxis_title=feature,
                    yaxis_title=target,
                    showlegend=False
                )
                st.plotly_chart(fig)
    else:
        st.warning("Please upload a CSV file.")

elif options == "Preprocessing & Data Manipulation":
    # Upload data
    st.header("Upload Data")
    # Ask the user to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        # Ask the user to select the required columns
        df = pd.read_csv(uploaded_file)
    
        # Handling Missing values
        st.subheader("Handling Missing Values")
        processed_df = pd.DataFrame()  
        selected_columns = st.multiselect("Select columns", df.columns.tolist())
        columns_to_process = selected_columns.copy()
        fill_methods = {}
        
        for col in selected_columns:
            if df[col].isnull().sum() <= 0:
                columns_to_process.remove(col)
                st.info(f"No Missing Values in Column: {col}")
            else:
                fill_methods[col] = st.selectbox(f"Select fill method for column {col}", ["No Method Selected","Drop Null Values", "Fill with mean", "Fill with median", "Fill with mode", "Backward fill", "Forward fill", "Fill with max", "Fill with min"])
        
        if len(columns_to_process) > 0:
            if st.checkbox("Fill Missing Values"):
                missing_cols = []
                for col in columns_to_process:
                    fill_method = fill_methods[col]
                    if fill_method == "No Method Selected":
                        missing_cols.append(col)
                    elif fill_method == "Drop Null Values":
                        df = df.dropna(subset=[col])
                    elif fill_method == "Fill with mean":
                        if df[col].dtype == "object":
                            st.error(f"Inappropriate way to fill missing values in column {col}. Please choose another method.")
                        else:
                            df[col].fillna(df[col].mean(), inplace=True)
                    elif fill_method == "Fill with median":
                        if df[col].dtype == "object":
                            st.error(f"Inappropriate way to fill missing values in column {col}. Please choose another method.")
                        else:
                            df[col].fillna(df[col].median(), inplace=True)
                    elif fill_method == "Fill with mode":
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif fill_method == "Backward fill":
                        df[col].fillna(method='bfill', inplace=True)
                    elif fill_method == "Forward fill":
                        df[col].fillna(method='ffill', inplace=True)
                    elif fill_method == "Fill with max":
                        if df[col].dtype == "object":
                            st.error(f"Inappropriate way to fill missing values in column {col}. Please choose another method.")
                        else:
                            df[col].fillna(df[col].max(), inplace=True)
                    elif fill_method == "Fill with min":
                        if df[col].dtype == "object":
                            st.error(f"Inappropriate way to fill missing values in column {col}. Please choose another method.")
                        else:
                            df[col].fillna(df[col].min(), inplace=True)
                    
                if len(missing_cols) > 0:
                    st.warning(f"Please select fill method for the following columns: {', '.join(missing_cols)}")
                else:
                    st.success("Missing values filled successfully!")
            
        processed_df = df[selected_columns].copy()
        
        st.subheader("Processed Data")
        st.dataframe(processed_df)
            
            
        # Label the categorical columns
        st.subheader("Labeling Categorical Columns")
        categorical_cols = processed_df.select_dtypes(include="object").columns.tolist()

        if len(categorical_cols) > 0:
            selected_cols = st.multiselect("Select columns to label", categorical_cols)
            remaining_cols = [col for col in categorical_cols if col not in selected_cols]

            for col in selected_cols:
                st.write(f"Processing column: {col}")
                encoding_method = st.selectbox(f"Select encoding method for column {col}", ["No Method Selected", "Label Encoder","OneHot Encoding"])

                if encoding_method == "No Method Selected":
                    st.warning(f"No encoding method selected for column {col}. Please choose another method.")
                elif encoding_method == "Label Encoder":
                    # Apply Label Encoder
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col])

                elif encoding_method == "OneHot Encoding":
                # Apply OneHot Encoding
                    encoded_df = pd.get_dummies(processed_df[col],columns=list(col),prefix=col)
                # Convert boolean columns to integer (0/1)
                    encoded_df = encoded_df.astype(int)
                    processed_df = pd.concat([processed_df,encoded_df],axis=1)
                    processed_df.drop([col],axis=1,inplace=True)
                
            if st.checkbox("Label the Categorical Columns"):
                if len(remaining_cols) > 0:
                    st.warning(f"Please select fill method for the following columns: {', '.join(remaining_cols)}")
                else:
                    st.success("Categorical Columns Labeled successfully!")
                    st.dataframe(processed_df)  
        else:
            st.info("No categorical columns found in the selected data.")

            processed_df = processed_df.copy()
        
        # Column selection
        st.subheader("Remove Duplicates")
        selected_columns = st.multiselect("Select columns", processed_df.columns.tolist())

        if st.checkbox("Remove Duplicates"):
            if len(selected_columns)>0:
            # Remove duplicates based on selected columns
                processed_df.drop_duplicates(subset=selected_columns, keep='first', inplace=True)
                # Display the deduplicated DataFrame
                st.success("Duplicated values removed Successfully!")
                st.dataframe(processed_df)
                processed_df = processed_df.copy()
            else:
                st.info("You didn't select any column")
                st.dataframe(processed_df)
                processed_df = processed_df.copy()
            
        st.subheader("Delete Columns")
        columns_to_be_deleted = processed_df.columns.to_list()
        selected_cols = st.multiselect("Please select the un-needed columns",columns_to_be_deleted)

        if st.checkbox("Apply Deleting Columns"):
            if len(selected_cols)>0:
                processed_df = processed_df.drop(selected_cols,axis=1)
                st.success("Selected Columns Deleted Successfully!")
                st.dataframe(processed_df)
                processed_df = processed_df.copy()
            else:
                st.info("You didn't select any column")
                st.dataframe(processed_df)
                processed_df = processed_df.copy()

        st.subheader("Rename Columns")
        columns_to_rename = processed_df.columns.to_list()
        selected_cols = st.multiselect("Select Column you need to rename",columns_to_rename)

        if len(selected_cols) > 0:
            rename_mapping = {}
            for col in selected_cols:
                new_name = st.text_input(f"Enter a new name for {col}", col)
                rename_mapping[col] = new_name

            if st.checkbox("Rename Columns"):
                processed_df.rename(columns=rename_mapping, inplace=True)
                processed_df = processed_df.copy()
                st.success("Selected Columns Renamed Successfully!")
                st.dataframe(processed_df)

        if st.checkbox("Download your new Data"):
            st.success("Data Processed Successfully,You can Download Your Processed Data By Clicking On the Below Button")
            # Display the button and dataframe
            if st.button('Download CSV'):
                st.markdown(download_data_as_csv(processed_df), unsafe_allow_html=True)
    else:
        st.warning("Please upload a CSV file.")

elif options == "Feature Selection":
    # Upload data
    st.header("Upload Data")
    data_file = st.file_uploader("Upload CSV file", type=["csv"])
    if data_file is not None:
        data = pd.read_csv(data_file)
        st.write("Data preview:")
        st.dataframe(data.head())

        # Select features and target variables
        st.header("Select Features and Target")
        features = st.multiselect("Select features", data.columns)
        target = st.selectbox("Select target variable", data.columns)

        if st.checkbox("Apply Feature Selection"):
            # Select feature selection method
            st.header("Select Feature Selection Method")
            method = st.selectbox("Select method",["No Method Selected","Chi-square", "ANOVA", "Correlation", "PCA"])
            
            # Apply feature selection method
            if method == "No Method Selected":
                st.warning("You didn't select any method")
            elif method == "Chi-square":
                k = st.number_input("Select number of features", 1, len(features), 1)
                selected_features, new_data = chi_square_feature_selection(data, features, target, k)
            elif method == "ANOVA":
                k = st.number_input("Select number of features", 1, len(features), 1)
                selected_features, new_data = anova_feature_selection(data, features, target, k)
            elif method == "Correlation":
                threshold = st.slider("Select correlation threshold", -1.0, 1.0, 0.5, 0.1)
                selected_features, new_data = correlation_feature_selection(data, features, target, threshold)
                fig ,ax = plt.subplots(figsize = (8,6))
                #mask = np.triu(np.ones_like(data.corr(), dtype = bool))
                #sns.heatmap(data.corr(), vmin=-1, vmax=1, mask = mask, annot=True, cmap='BrBG')
                plt.title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
                sns.heatmap(data.corr()[[target]].sort_values(by=target,ascending=False),vmin=-1, vmax=1, annot=True, cmap='BrBG')
                plt.title(f'Features Correlating with {target}', fontdict={'fontsize':18}, pad=16)
                st.pyplot(fig)
                #st.pyplot(ax)
            elif method == "PCA":
                n_components = st.number_input("Select number of components", 1, len(features), 1)
                selected_features, new_data = pca_feature_selection(data, features, n_components)

            if st.checkbox("Display Selected Features"):
                # Display selected features and new data
                st.subheader("Selected Features")
                st.write(selected_features)
            else:
                st.warning("Please Click on the above checkbox to display the selected features")
            if st.checkbox("New Data"):        
                st.subheader("New Data")
                st.dataframe(new_data)
            else:
                st.warning("Please Click on the above checkbox to display your new data")
                
            if st.checkbox("Download your new Data"):
                st.success("Features Selection Performed Successfully,You can Download Your New Data By Clicking On the Below Button")
                # Display the button and dataframe
                if st.button('Download CSV'):
                    st.markdown(download_data_as_csv(new_data), unsafe_allow_html=True)

    else:
        st.warning("Please upload a CSV file.")

elif options == "ML Models":
    # Upload data
    st.header("Upload Data")
    data_file = st.file_uploader("Upload CSV file", type=["csv"])
    if data_file is not None:
        data = pd.read_csv(data_file)
        st.write("Data preview:")
        st.dataframe(data.head())

    #Ask the user to select the features and target variable
        feature_cols = st.multiselect("Select Features", data.columns)
        target_col = st.selectbox("Select the target variable", data.columns)

        test_size = st.number_input("Enter Test Size (0 to 1)", min_value=0.0, max_value=1.0, value=0.2)
        random_state = st.number_input("Enter Random State", min_value=0, max_value=100, value=42)

        X_train, X_test, y_train, y_test = split_data(data, feature_cols, target_col, test_size, random_state)

        # Standardization
        st.subheader("Standardization")
        standardize_col = X_train.columns.to_list()
        selected_cols = st.multiselect("Select column to standardize", standardize_col)

        if len(selected_cols)> 0:
            for col in selected_cols:
                st.write(f"Processing column: {col}")
                standardize_method = st.selectbox(f"Select standardization method for column {col}", ["No Method Selected", "Standard Scaler", "Min-Max Scaler"])
                if standardize_method == "No Method Selected":
                    st.warning(f"No standardization method selected for column {col}. Please choose another method.")
                elif standardize_method == "Standard Scaler":
                    # Apply Standard Scaler method
                    scaler = StandardScaler()
                    X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
                    X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
                elif standardize_method == "Min-Max Scaler":
                    # Apply Min-Max Scaler method
                    scaler = MinMaxScaler()
                    X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
                    X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
       
            if st.checkbox("Perform Standardization"):
                st.success("Standardization performed successfully!")
                X_train = X_train.copy()
                X_test = X_test.copy()
                st.subheader("Train Data Sample after Standardization")
                st.dataframe(X_train.sample(3))
                st.subheader("Test Data Sample after Standardization")
                st.dataframe(X_test.sample(3))
                
                if st.checkbox("Download your Scaling method"):
                    if standardize_method == "No Method Selected":
                        st.warning("You aren't able to download as you choosed a wrong selection in the standardize method")
                    else:
                        with open('scaler.pickle', 'wb') as f:
                            pickle.dump(scaler, f)
                        with open('scaler.pickle', 'rb') as f:
                            model_bytes = f.read()
                        st.download_button(
                            label="Download Scaler",
                            data=model_bytes,
                            file_name='scaler.pickle',
                            mime='application/octet-stream')


                
        else:
            st.info("You didn't choose any column to Standardize")
            X_train = X_train.copy()
            X_test = X_test.copy()
            
        # Ask the user about the problem type
        problem_type = st.selectbox("Select the problem type", ["No problem type selected","Regression", "Classification"])

        if problem_type == "No problem type selected":
            st.warning("Please select the problem type")
        elif problem_type == "Regression":
            # Choose regression algorithms
            regression_algorithms = ["No Model Selected","Linear Regression","Lasso Regression","Ridge Regression","Random Forest Regressor","XGBOOST Regressor","SVM Regressor"]
            selected_algorithm = st.selectbox("Select a regression algorithm",regression_algorithms)

            if selected_algorithm == "No Model Selected":
                st.warning("Please Select Your Requiered Model!")
            elif selected_algorithm == "Linear Regression":
                linear_regression(X_train, X_test, y_train, y_test)
            elif selected_algorithm == "Lasso Regression":
                lasso_regression(X_train, X_test, y_train, y_test)
            elif selected_algorithm == "Ridge Regression":
                ridge_regression(X_train, X_test, y_train, y_test)
            elif selected_algorithm == "Random Forest Regressor":
                Random_Forest_regression(X_train, X_test, y_train, y_test)
            elif selected_algorithm == "XGBOOST Regressor":
                XGBOOST_regression(X_train, X_test, y_train, y_test)
            elif selected_algorithm == "SVM Regressor" :
                SVM_regression(X_train, X_test, y_train, y_test)
            
        elif problem_type == "Classification":
        # Choose classification algorithms
            classification_algorithms = ["No Model Selected","Logistic Regression","SVM Classifier","Random Forest Classifier","XGBOOST Classifier","Decision Tree"]
            selected_algorithm = st.selectbox("Select a classification algorithm",classification_algorithms )
            
            if selected_algorithm == "No Model Selected":
                st.warning("Please Select Your Requiered Model!")
            elif selected_algorithm == "Logistic Regression":
                logistic_regression(X_train, X_test, y_train, y_test)
            elif selected_algorithm == "SVM Classifier":
                SVM_classifier(X_train, X_test, y_train, y_test)
            elif selected_algorithm == "Random Forest Classifier":
                random_forest_classifier(X_train, X_test, y_train, y_test),
            elif selected_algorithm == "XGBOOST Classifier":
                XGBOOST_classifier(X_train, X_test, y_train, y_test)
            elif selected_algorithm == "Decision Tree":
                Decision_tree(X_train, X_test, y_train, y_test)

        # Hyperparameter tuning
        if st.checkbox("Apply Hyperparameter Tuning"):
            if problem_type == "No problem type selected":
                st.warning("Please select the problem type")
            elif problem_type == "Regression":
                if selected_algorithm == "No Model Selected":
                    st.warning("Please Select Your Required Model!")
                elif selected_algorithm == "Linear Regression":
                    param_grid = {'fit_intercept': [True, False],'copy_X': [True, False],"positive":[True, False]}
                    search_type = st.selectbox("Select a search type", ["No search type selected!","Grid Search", "Random Search"])
                    if search_type == "No search type selected!":
                        st.warning("Please select the requiered search type!")
                    else:
                        model = LinearRegression()
                        results_df = hyperparameter_tuning(model, param_grid, search_type, X_train, X_test, y_train, y_test)
                elif selected_algorithm == "Lasso Regression":
                    param_grid = {'alpha': [0.1, 1.0, 10.0]}
                    search_type = st.selectbox("Select a search type", ["No search type selected!","Grid Search", "Random Search"])
                    if search_type == "No search type selected!":
                        st.warning("Please select the requiered search type!")
                    else:
                        model = Lasso()
                        results_df = hyperparameter_tuning(model, param_grid, search_type, X_train, X_test, y_train, y_test)
                elif selected_algorithm == "Ridge Regression":
                    param_grid = {'alpha': [0.1, 1.0, 10.0]}
                    search_type = st.selectbox("Select a search type", ["No search type selected!","Grid Search", "Random Search"])
                    if search_type == "No search type selected!":
                        st.warning("Please select the requiered search type!")
                    else:
                        model = Ridge()
                        results_df = hyperparameter_tuning(model, param_grid, search_type, X_train, X_test, y_train, y_test)
                elif selected_algorithm == "Random Forest Regressor":
                    # Define the parameter grid for Random Forest Regressor
                    param_grid = {
                        'n_estimators': [10, 50, 100],
                        'max_depth': [None, 5, 10]
                    }
                    search_type = st.selectbox("Select a search type", ["No search type selected!","Grid Search", "Random Search"])
                    if search_type == "No search type selected!":
                        st.warning("Please select the requiered search type!")
                    else: 
                        model = RandomForestRegressor()
                        results_df = hyperparameter_tuning(model, param_grid, search_type, X_train, X_test, y_train, y_test)
                elif selected_algorithm == "XGBOOST Regressor":
                    # Define the parameter grid for XGBoost Regressor
                    param_grid = {
                        'learning_rate': [0.1, 0.01],
                        'max_depth': [3, 5, 7]
                    }
                    search_type = st.selectbox("Select a search type", ["No search type selected!","Grid Search", "Random Search"])
                    if search_type == "No search type selected!":
                        st.warning("Please select the requiered search type!")
                    else: 
                        model = XGBRegressor()
                        results_df = hyperparameter_tuning(model, param_grid, search_type, X_train, X_test, y_train, y_test)
                elif selected_algorithm == "SVM Regressor" :
                    # Define the parameter grid for SVM Regressor
                    param_grid = {
                        'C': [1, 10, 100],
                        'kernel': ['linear', 'rbf']
                    }
                    search_type = st.selectbox("Select a search type", ["No search type selected!","Grid Search", "Random Search"])
                    if search_type == "No search type selected!":
                        st.warning("Please select the requiered search type!")
                    else:
                        model = SVR()
                        results_df = hyperparameter_tuning(model, param_grid, search_type, X_train, X_test, y_train, y_test)

            elif problem_type == "Classification":
                if selected_algorithm == "No Model Selected":
                    st.warning("Please Select Your Required Model!")
                elif selected_algorithm == "Logistic Regression":
                    # Define the parameter grid for Logistic Regression
                    param_grid = {
                        'C': [1, 10, 100],
                        'penalty': ['l1', 'l2']
                    }
                    search_type = st.selectbox("Select a search type", ["No search type selected!","Grid Search", "Random Search"])
                    if search_type == "No search type selected!":
                        st.warning("Please select the requiered search type!")
                    else:
                        model = LogisticRegression()
                        results_df = hyperparameter_tuning(model, param_grid, search_type, X_train, X_test, y_train, y_test)
                elif selected_algorithm == "SVM Classifier":
                    # Define the parameter grid for SVM Classifier
                    param_grid = {
                        'C': [1, 10, 100],
                        'kernel': ['linear', 'rbf']
                    }
                    search_type = st.selectbox("Select a search type", ["No search type selected!","Grid Search", "Random Search"])
                    if search_type == "No search type selected!":
                        st.warning("Please select the requiered search type!")
                    else:
                        model = SVC()
                        results_df = hyperparameter_tuning(model, param_grid, search_type, X_train, X_test, y_train, y_test)
                elif selected_algorithm == "Random Forest Classifier":
                    # Define the parameter grid for Random Forest Classifier
                    param_grid = {
                        'n_estimators': [10, 50, 100],
                        'max_depth': [None, 5, 10]
                    }
                    search_type = st.selectbox("Select a search type", ["No search type selected!","Grid Search", "Random Search"])
                    if search_type == "No search type selected!":
                        st.warning("Please select the requiered search type!")
                    else:
                        model = RandomForestClassifier()
                        results_df = hyperparameter_tuning(model, param_grid, search_type, X_train, X_test, y_train, y_test)
                elif selected_algorithm == "XGBOOST Classifier":
                    # Define the parameter grid for XGBoost Classifier
                    param_grid = {
                        'learning_rate': [0.1, 0.01],
                        'max_depth': [3, 5, 7]
                    }
                    search_type = st.selectbox("Select a search type", ["No search type selected!","Grid Search", "Random Search"])
                    if search_type == "No search type selected!":
                        st.warning("Please select the requiered search type!")
                    else:
                        model = XGBClassifier()
                        results_df = hyperparameter_tuning(model, param_grid, search_type, X_train, X_test, y_train, y_test)
                elif selected_algorithm == "Decision Tree":
                    # Define the parameter grid for Decision Tree
                    param_grid = {
                        'max_depth': [None, 5, 10]}
                    search_type = st.selectbox("Select a search type", ["No search type selected!","Grid Search", "Random Search"])
                    if search_type == "No search type selected!":
                        st.warning("Please select the requiered search type!")
                    else:
                        model = DecisionTreeClassifier()
                        results_df = hyperparameter_tuning(model, param_grid, search_type, X_train, X_test, y_train, y_test)
                        
    else:
        st.warning("Please upload a CSV file.")        
