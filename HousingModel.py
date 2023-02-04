import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

MAX_ITERATIONS = 10000

train_csv = pd.read_csv("/content/sample_data/california_housing_test.csv")
test_csv = pd.read_csv("/content/sample_data/california_housing_train.csv")

housing_dataset = train_csv.append(test_csv)

x = housing_dataset.drop(columns= ["median_house_value"])
y = housing_dataset["median_house_value"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

housing_model_IPR = HistGradientBoostingRegressor(max_iter=MAX_ITERATIONS, random_state=1)
housing_model = housing_model_IPR.fit(x_train, y_train)

y_predict = housing_model.predict(x)
y_predict[y_predict < 0] = 0

print("accuracy: " + str(housing_model.score(x, y)))
print("r2 score: " + str(r2_score(y_predict, housing_model.predict(x))))

pd_input = input(str("do u want to see predicted dataset?"))
if pd_input.lower() == "yes":
    predicted_database = pd.DataFrame({"value": y,"predicted value": y_predict})
    predicted_database.reset_index(drop=True, inplace=True)
    predicted_database.to_csv("predicted_california_housing.csv", index=False, encoding="utf-8")