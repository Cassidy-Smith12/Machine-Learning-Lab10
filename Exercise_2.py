import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
column_names = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'groove', 'class']
data = pd.read_csv(url, sep='\t+', header=None, names=column_names, engine='python')

# Separate features and target
X = data.drop('class', axis=1)
y = data['class']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build and evaluate MLPClassifier with three architectures
architectures = [
    {'hidden_layer_sizes': (50,), 'max_iter': 500, 'activation': 'relu'},
    {'hidden_layer_sizes': (100, 50), 'max_iter': 500, 'activation': 'tanh'},
    {'hidden_layer_sizes': (150, 100, 50), 'max_iter': 500, 'activation': 'relu'}
]

for arch in architectures:
    mlp = MLPClassifier(**arch, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Architecture: {arch}, Accuracy: {acc:.2f}")
