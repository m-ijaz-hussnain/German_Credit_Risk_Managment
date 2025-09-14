import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from tensorflow import keras
from keras import layers, models
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

import os

def load_data(file_path):
    df = pd.read_excel(file_path)
    print('Data loaded Successfully')

def preprocess_data(df):
    original_data = df.copy()    # copy data

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        print("\nDropped 'Unnamed: 0' column.")
    
    # numerical and catagorical features
    numerical_features = ['Age', 'Credit amount', 'Duration']
    categorical_features = [
        'Sex', 'Job', 'Housing', 'Saving accounts',
        'Checking account', 'Purpose'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            # 'num': Apply StandardScaler for numerical features (scaling to mean 0, variance 1)
            ('num', StandardScaler(), numerical_features),
            # 'cat': Apply OneHotEncoder for categorical features (converts categories to numerical columns)
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    # Apply preprocessing
    X = df.drop(columns=['Risk'])
    y = df['Risk']  # Assuming 'Risk' is the target variable
   # Encode good or bad into 0 or 1 in risk column
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Encoded 'Risk' column. Original classes: {label_encoder.classes_}")

    x_preprocessor = preprocessor.fit_transform(X)

    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    # Combine original numerical feature names with new one-hot encoded feature names
    all_feature_names = numerical_features + list(cat_feature_names)
    X_processed_df = pd.DataFrame(x_preprocessor, columns=all_feature_names)

    output_file_name_processed_features = 'Credit Risk Managment\german_credit_data_preprocessed_features.xlsx'
    X_processed_df.to_excel(output_file_name_processed_features, index=False)
    print(f"\nProcessed features saved to '{output_file_name_processed_features}'")

    output_file_name_encoded_target = 'Credit Risk Managment\german_credit_data_encoded_target.xlsx'
    pd.DataFrame(y_encoded, columns=['Risk_Encoded']).to_excel(output_file_name_encoded_target, index=False)
    print(f"Encoded target saved to '{output_file_name_encoded_target}'")

def apply_models():
    # Load preprocessed data
    X = pd.read_excel('Credit Risk Managment\\german_credit_data_preprocessed_features.xlsx')
    y = pd.read_excel('Credit Risk Managment\\german_credit_data_encoded_target.xlsx')['Risk_Encoded']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    dt_predictions = dt_classifier.predict(X_test)
    print("Decision Tree Classifier Accuracy:", accuracy_score(y_test, dt_predictions))

    # Train a Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)
    rf_predictions = rf_classifier.predict(X_test)
    print("Random Forest Classifier Accuracy:", accuracy_score(y_test, rf_predictions))

    # Train linear regression model
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(X_train, y_train)
    linear_regression_predictions = linear_regression_model.predict(X_test)
    print("Linear Regression Model Accuracy:", linear_regression_model.score(X_test, y_test))

    # Train neural networks
    neural_network_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Assuming binary classification
    ])
    neural_network_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    neural_network_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    nn_predictions = (neural_network_model.predict(X_test) > 0.5).astype("int32")
    print("Neural Network Model Accuracy:", accuracy_score(y_test, nn_predictions))

    # Save the trained models
    joblib.dump(dt_classifier, r'Credit Risk Managment\\decision_tree_classifier.pkl')
    joblib.dump(rf_classifier, r'Credit Risk Managment\\random_forest_classifier.pkl')
    joblib.dump(linear_regression_model, r'Credit Risk Managment\\linear_regression_model.pkl')
    neural_network_model.save(r'Credit Risk Managment\\neural_network_model.h5')
    print("Models saved successfully.")


if __name__ == "__main__":
    file_path = 'Credit Risk Managment\german_credit_data.xlsx'
    load_data(file_path)
    df = pd.read_excel(file_path)
    preprocess_data(df)
    print("Data preprocessing completed successfully.")
    apply_models()