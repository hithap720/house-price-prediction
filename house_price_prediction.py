import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000],
    'Price': [5000000, 6000000, 7500000, 9000000, 10000000]
}

df = pd.DataFrame(data)

X = df[['Area']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

area = [[1600]]
predicted_price = model.predict(area)

print("Predicted House Price:", predicted_price[0])