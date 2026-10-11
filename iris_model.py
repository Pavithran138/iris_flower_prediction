import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("iris.csv")
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("--- Iris Flower Prediction ---")
try:
    sepal_length = float(input("Enter Sepal Length (cm): "))
    sepal_width = float(input("Enter Sepal Width (cm): "))
    petal_length = float(input("Enter Petal Length (cm): "))
    petal_width = float(input("Enter Petal Width (cm): "))
    user_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)
    prediction = model.predict(user_data)
    
    print(f"\n=> The predicted flower species is: {prediction[0].capitalize()}")
except ValueError:
    print("Invalid input. Please enter numerical values.")