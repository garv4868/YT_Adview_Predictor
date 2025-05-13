import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn import metrics, linear_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
from keras.models import Sequential, load_model # type: ignore
from keras.layers import Dense # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.losses import mean_squared_error # type: ignore
import re

# Function to clean and preprocess data
def preprocess_data(data):
    # Remove videos with adview greater than 2000000 as outliers
    if 'adview' in data.columns:
        data = data[data["adview"] < 2000000]
    
    # Removing character "F" present in data
    for col in ['views', 'likes', 'dislikes', 'comment']:
        data = data[data[col] != 'F']
    
    # Convert values to numeric
    for col in ['views', 'likes', 'dislikes', 'comment']:
        data[col] = pd.to_numeric(data[col])
    
    if 'adview' in data.columns:
        data['adview'] = pd.to_numeric(data['adview'])
    
    # Encode categorical features
    category = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
    data["category"] = data["category"].map(category)
    
    # Encode 'vidid' and 'published' features
    data['vidid'] = LabelEncoder().fit_transform(data['vidid'])
    data['published'] = LabelEncoder().fit_transform(data['published'])
    
    # Convert duration to seconds
    data['duration'] = data['duration'].apply(convert_duration)
    
    return data

# Function to convert duration to seconds
def convert_duration(duration):
    match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', duration)
    hours = int(match.group(1)[:-1]) if match.group(1) else 0
    minutes = int(match.group(2)[:-1]) if match.group(2) else 0
    seconds = int(match.group(3)[:-1]) if match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds

# Function to evaluate models
def print_error(X_test, y_test, model):
    prediction = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, prediction)
    mse = metrics.mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    return mae, mse, rmse

# Importing training data
data_train = pd.read_csv('train_AdView.csv')
data_train = preprocess_data(data_train)

# Plot heatmap for training data
plt.figure(figsize=(10, 8))
sns.heatmap(data_train.corr(), cmap='coolwarm', annot=True)
plt.show()

# Splitting features and target variable
Y_train = data_train.pop('adview')
data_train = data_train.drop(["vidid"], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_train, Y_train, test_size=0.2, random_state=42)

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate different models
models = {
    'Linear Regression': linear_model.LinearRegression(),
    'Support Vector Regressor': SVR(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=200, max_depth=25, min_samples_split=15, min_samples_leaf=2)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f'{name} Errors:')
    results[name] = print_error(X_test, y_test, model)
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_youtubeadview.pkl")

# Train and evaluate Artificial Neural Network
ann = Sequential([
    Dense(6, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(6, activation='relu'),
    Dense(1)
])
ann.compile(optimizer=Adam(), loss=mean_squared_error, metrics=['mean_squared_error'])
ann.fit(X_train, y_train, epochs=100)
ann.summary()
print('Artificial Neural Network Errors:')
results['Artificial Neural Network'] = print_error(X_test, y_test, ann)
ann.save("artificial_neural_network_youtubeadview.keras")

# Selecting the best model based on RMSE
best_model_name = min(results, key=lambda k: results[k][2])
print(f'The best model is: {best_model_name} with RMSE: {results[best_model_name][2]}')

# Import and preprocess testing data
data_test = pd.read_csv("test.csv")
data_test = preprocess_data(data_test)
data_test = data_test.drop(["vidid"], axis=1)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data_test.corr(), cmap='coolwarm', annot=True)
plt.show()

# Normalize testing data
X_test = scaler.transform(data_test)

# Load the best model and make predictions
if best_model_name == 'Artificial Neural Network':
    best_model = load_model("artificial_neural_network_youtubeadview.keras")
else:
    best_model = joblib.load(f"{best_model_name.lower().replace(' ', '_')}_youtubeadview.pkl")

predictions = best_model.predict(X_test)
predictions = pd.DataFrame(predictions, columns=["Adview"])

# Save predictions to CSV
predictions.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")