import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Split the data into features and target
X = data[['Age', 'BMI', 'BloodPressure', 'Insulin']]  # Features
y = data['Outcome']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Scale training data
X_test = sc.transform(X_test)        # Scale test data

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save the model and the scaler
pickle.dump(model, open('classifier.pkl', 'wb'))
pickle.dump(sc, open('sc.pkl', 'wb'))  # Save the scaler for later use in Flask app


# # Take user input for prediction
# print("\nEnter the following details for diabetes prediction:")
# age = float(input("Age: "))
# bmi = float(input("BMI (Body Mass Index): "))
# blood_pressure = float(input("Blood Pressure: "))
# insulin = float(input("Insulin Level: "))

# # Prepare the input for prediction
# user_data = [[age, bmi, blood_pressure, insulin]]

# # Predict diabetes status
# prediction = model.predict(user_data)[0]

# # Output the prediction
# if prediction == 1:
#     print("Diabetes Prediction: Yes")
# else:
#     print("Diabetes Prediction: No")