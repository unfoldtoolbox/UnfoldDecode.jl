from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the Iris dataset
iris = load_iris()
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [[-2, 1], [-1, 1], [-1, 2], [1, 10], [1, 20], [2, 10]]
print(X)
print(y)

# do not split the dataset into training and testing sets
# Create and train the RidgeRegressor model
model = Ridge(alpha=1.0)
model.fit(y, X)

# get the projection array G
G = model.coef_
print(G)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the RidgeRegressor model
# model = Ridge(alpha=1.0)
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Calculate the mean squared error
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)
