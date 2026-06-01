import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# load kaggle dataset
df = pd.read_csv("train_house_price.csv")

# keep only required columns
df = df[[
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "OverallQual",
    "GarageCars",
    "YearBuilt",
    "SalePrice"
]]

# remove missing values if any
df = df.dropna()

# features and target
X = df[[
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "OverallQual",
    "GarageCars",
    "YearBuilt",
]]

y = df["SalePrice"]

# split data
xtrain, xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# create model
model = LinearRegression()

# train model
model.fit(xtrain, ytrain)

# predictions
pred = model.predict(xtest)

# evaluation
error = mean_absolute_error(ytest, pred)

print("Mean Absolute Error:", error)

# save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("model saved")