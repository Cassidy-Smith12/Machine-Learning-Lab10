import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

print(f'Description: {diabetes.DESCR}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#different MLP architectures
mlp_models = {
    "Model 1": MLPRegressor(hidden_layer_sizes=(64), activation='tanh', solver='adam', max_iter=10000, random_state=42),
    "Model 2": MLPRegressor(hidden_layer_sizes=(64, 32), activation='logistic', solver='adam', max_iter=10000, random_state=42),
    "Model 3": MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=10000, random_state=42)
}

results = {}
for name, model in mlp_models.items():
    
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"MSE": mse, "R2 Score": r2}
    print(f"\n{name}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

plt.figure(figsize=(10, 5))
for name, model in mlp_models.items():
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, label=name)
    
plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Target Values")
plt.title("Comparison of Different MLP Architectures")
plt.legend()
plt.grid()
plt.show()
