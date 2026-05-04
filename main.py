import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Scores": [30, 35, 50, 55, 60, 65, 70, 80]
}

df = pd.DataFrame(data)

# Step 2: Prepare data
X = df[["Hours"]]
y = df["Scores"]

# Step 3: Train model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict
hours = float(input("Enter study hours: "))
predicted_score = model.predict([[hours]])

print(f"Predicted Score: {predicted_score[0]:.2f}")

# Step 5: Plot graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.title("Study Hours vs Score")
plt.show()