# iris_classifier.py

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_data():
    iris = datasets.load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['species'] = iris.target
    return data, iris.target_names

def preprocess_data(data):
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Labels
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy of the model: {accuracy * 100:.2f}%")

def main():
    data, species_names = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model, scaler = train_model(X_train, y_train)
    evaluate_model(model, scaler, X_test, y_test)

if __name__ == "__main__":
    main()