import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# EDA

## Importar datos

total_data=pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv")

total_data.to_csv("../data/raw/total_data.csv", index = False)

## Eliminar duplicados

total_data = total_data.drop_duplicates().reset_index(drop = True)

## Análisis de variables univariante

### Categóricas

fig, axis = plt.subplots(2, 2, figsize = (16, 8))

sns.histplot(ax = axis[0, 0], data = total_data, x = "sex")
sns.histplot(ax = axis[0, 1], data = total_data, x = "smoker")
sns.histplot(ax = axis[1, 0], data = total_data, x = "region")

fig.delaxes(axis[1, 1])

plt.tight_layout()

plt.show()

### Numéricas

fig, axis = plt.subplots(4, 2, figsize=(14, 10), gridspec_kw={"height_ratios": [6, 1] * 2})

sns.histplot(ax=axis[0, 0], data=total_data, x="age")
sns.boxplot(ax=axis[1, 0], data=total_data, x="age")

sns.histplot(ax=axis[0, 1], data=total_data, x="bmi")
sns.boxplot(ax=axis[1, 1], data=total_data, x="bmi")

sns.histplot(ax=axis[2, 0], data=total_data, x="children")
sns.boxplot(ax=axis[3, 0], data=total_data, x="children")

sns.histplot(ax=axis[2, 1], data=total_data, x="charges")
sns.boxplot(ax=axis[3, 1], data=total_data, x="charges")

plt.tight_layout()

plt.show()

## Análisis de variables multivariante

### Numérico - numérico

fig, axis = plt.subplots(4, 2, figsize = (8, 12))

sns.regplot(ax = axis[0, 0], data = total_data, x = "bmi", y = "charges")
sns.heatmap(total_data[["charges", "bmi"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)
sns.regplot(ax = axis[0, 1], data = total_data, x = "age", y = "charges").set(ylabel=None)
sns.heatmap(total_data[["charges", "age"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])
sns.regplot(ax = axis[2, 0], data = total_data, x = "children", y = "charges")
sns.heatmap(total_data[["charges", "children"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0], cbar = False)

plt.delaxes(axis[2,1])
plt.delaxes(axis[3,1])

plt.tight_layout()

plt.show()

#### Age- Children

sns.regplot(ax = axis[0,0], data = total_data, x = "children", y = "age")
sns.heatmap(total_data[["age", "children"]].corr(), annot = True, fmt = ".2f", ax = axis[1,0])

sns.regplot(ax = axis[0,1], data = total_data, x = "bmi", y = "age")
sns.heatmap(total_data[["age", "bmi"]].corr(), annot = True, fmt = ".2f", ax = axis[1,1])

plt.tight_layout()

plt.show()

### Categórico - categórico

total_data["sex_n"] = pd.factorize(total_data["sex"])[0]
total_data["smoker_n"] = total_data["smoker"].map({"yes": 1, "no": 0})
total_data["region_n"] = pd.factorize(total_data["region"])[0]

fig, axis = plt.subplots(figsize = (10, 6))

sns.heatmap(total_data[["sex_n", "smoker_n", "region_n", "charges"]].corr(), annot=True, fmt=".2f")

plt.tight_layout()

plt.show()

columnas = [
    ("sex", "sex_n"),
    ("smoker", "smoker_n"),
    ("region", "region_n"),
]

transformation_rules = {}

for original_col, normalized_col in columnas:
    mapping = {
        row[original_col]: row[normalized_col]
        for _, row in total_data[[original_col, normalized_col]].drop_duplicates().iterrows()
    }
    transformation_rules[original_col] = mapping


with open("../models/transformation_rules.json", "w") as f:
    json.dump(transformation_rules, f, indent=4)

### Numérico - categórico

numericas_continuas = [
    "age", "bmi", "children", "charges"
]

categoricas_normalizadas = [
    "sex_n", "smoker_n", "region_n"
]

columnas_para_heatmap = numericas_continuas + categoricas_normalizadas

fig, axes = plt.subplots(figsize=(16, 10))

sns.heatmap(total_data[columnas_para_heatmap].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()

## Ingeniería de características

total_data_con_outliers = total_data.copy()
total_data_sin_outliers = total_data.copy()

def replace_outliers_from_column(column, df):
  column_stats = df[column].describe()
  column_iqr = column_stats["75%"] - column_stats["25%"]
  upper_limit = column_stats["75%"] + 1.5 * column_iqr
  lower_limit = column_stats["25%"] - 1.5 * column_iqr

  if lower_limit < 0:
    lower_limit = float(df[column].min())
  # Remove upper outliers
  df[column] = df[column].apply(lambda x: x if (x <= upper_limit) else upper_limit)
  # Remove lower outliers
  df[column] = df[column].apply(lambda x: x if (x >= lower_limit) else lower_limit)
  return df.copy(), [lower_limit, upper_limit]

outliers_dict = {}
for column in numericas_continuas:
  total_data_sin_outliers, limits_list = replace_outliers_from_column(column, total_data_sin_outliers)
  outliers_dict[column] = limits_list

with open("../models/outliers_replacement.json", "w") as f:
    json.dump(outliers_dict, f)

## Escalado de valores

num_variables = ["age", "bmi", "children", "sex_n", "smoker_n", "region_n"]

X_con_outliers = total_data_con_outliers[num_variables]
X_sin_outliers = total_data_sin_outliers[num_variables]
y = total_data_con_outliers["charges"]

X_train_con_outliers, X_test_con_outliers, y_train, y_test = train_test_split(X_con_outliers, y, test_size = 0.2, random_state = 42)
X_train_sin_outliers, X_test_sin_outliers = train_test_split(X_sin_outliers, test_size = 0.2, random_state = 42)


X_train_con_outliers.to_excel("../data/processed/X_train_con_outliers.xlsx", index = False)
X_train_sin_outliers.to_excel("../data/processed/X_train_sin_outliers.xlsx", index = False)
X_test_con_outliers.to_excel("../data/processed/X_test_con_outliers.xlsx", index = False)
X_test_sin_outliers.to_excel("../data/processed/X_test_sin_outliers.xlsx", index = False)
y_train.to_excel("../data/processed/y_train.xlsx", index = False)
y_test.to_excel("../data/processed/y_test.xlsx", index = False)

### Normalización

normalizador_con_outliers = StandardScaler()
normalizador_con_outliers.fit(X_train_con_outliers)

with open("../models/normalizador_con_outliers.pkl", "wb") as file:
    pickle.dump(normalizador_con_outliers,file)

X_train_con_outliers_norm = normalizador_con_outliers.transform(X_train_con_outliers)
X_train_con_outliers_norm = pd.DataFrame(X_train_con_outliers_norm, index = X_train_con_outliers.index, columns = num_variables)

X_test_con_outliers_norm = normalizador_con_outliers.transform(X_test_con_outliers)
X_test_con_outliers_norm = pd.DataFrame(X_test_con_outliers_norm, index = X_test_con_outliers.index, columns = num_variables)

X_train_con_outliers_norm.to_excel("../data/processed/X_train_con_outliers_norm.xlsx", index = False)
X_test_con_outliers_norm.to_excel("../data/processed/X_test_con_outliers_norm.xlsx", index = False)

normalizador_sin_outliers = StandardScaler()
normalizador_sin_outliers.fit(X_train_sin_outliers)

with open("../models/normalizador_sin_outliers.pkl", "wb") as file:
    pickle.dump(normalizador_sin_outliers,file)

X_train_sin_outliers_norm = normalizador_sin_outliers.transform(X_train_sin_outliers)
X_train_sin_outliers_norm = pd.DataFrame(X_train_sin_outliers_norm, index = X_train_sin_outliers.index, columns = num_variables)

X_test_sin_outliers_norm = normalizador_sin_outliers.transform(X_test_sin_outliers)
X_test_sin_outliers_norm = pd.DataFrame(X_test_sin_outliers_norm, index = X_test_sin_outliers.index, columns = num_variables)

X_train_sin_outliers_norm.to_excel("../data/processed/X_train_sin_outliers_norm.xlsx", index = False)
X_test_sin_outliers_norm.to_excel("../data/processed/X_test_sin_outliers_norm.xlsx", index = False)

X_train_con_outliers_norm.head()

### Min-max

min_max_con_outliers = MinMaxScaler()
min_max_con_outliers.fit(X_train_con_outliers)

with open("../models/min_max_con_outliers.pkl", "wb") as file:
    pickle.dump(min_max_con_outliers,file)

X_train_con_outliers_scal = min_max_con_outliers.transform(X_train_con_outliers)
X_train_con_outliers_scal = pd.DataFrame(X_train_con_outliers_scal, index = X_train_con_outliers.index, columns = num_variables)

X_test_con_outliers_scal = min_max_con_outliers.transform(X_test_con_outliers)
X_test_con_outliers_scal = pd.DataFrame(X_test_con_outliers_scal, index = X_test_con_outliers.index, columns = num_variables)

X_train_con_outliers_scal.to_excel("../data/processed/X_train_con_outliers_scal.xlsx", index = False)
X_test_con_outliers_scal.to_excel("../data/processed/X_test_con_outliers_scal.xlsx", index = False)

min_max_sin_outliers = MinMaxScaler()
min_max_sin_outliers.fit(X_train_sin_outliers)

with open("../models/min_max_sin_outliers.pkl", "wb") as file:
    pickle.dump(min_max_sin_outliers,file)

X_train_sin_outliers_scal = min_max_sin_outliers.transform(X_train_sin_outliers)
X_train_sin_outliers_scal = pd.DataFrame(X_train_sin_outliers_scal, index = X_train_sin_outliers.index, columns = num_variables)

X_test_sin_outliers_scal = min_max_sin_outliers.transform(X_test_sin_outliers)
X_test_sin_outliers_scal = pd.DataFrame(X_test_sin_outliers_scal, index = X_test_sin_outliers.index, columns = num_variables)

X_train_sin_outliers_scal.to_excel("../data/processed/X_train_sin_outliers_scal.xlsx", index = False)
X_test_sin_outliers_scal.to_excel("../data/processed/X_test_sin_outliers_scal.xlsx", index = False)

X_train_con_outliers_scal.head()

## Feature selection

selection_model = SelectKBest(f_regression, k = 4)
selection_model.fit(X_train_con_outliers_scal, y_train)

ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train_con_outliers_scal), columns = X_train_con_outliers_scal.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test_con_outliers_scal), columns = X_test_con_outliers_scal.columns.values[ix])

with open("../models/feature_selection_k_4.json", "w") as f:
    json.dump(X_train_sel.columns.tolist(), f)

X_train_sel.head()

X_train_sel["charges"] = y_train.values
X_test_sel["charges"] = y_test.values

X_train_sel.to_csv("../data/processed/clean_train.csv", index=False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index=False)

# Machine learning: Regresión lineal

BASE_PATH = "../data/processed"
TRAIN_PATHS = [
    "X_train_con_outliers.xlsx",
    "X_train_sin_outliers.xlsx",
    "X_train_con_outliers_norm.xlsx",
    "X_train_sin_outliers_norm.xlsx",
    "X_train_con_outliers_scal.xlsx",
    "X_train_sin_outliers_scal.xlsx"
]
TRAIN_DATASETS = []
for path in TRAIN_PATHS:
    TRAIN_DATASETS.append(
        pd.read_excel(f"{BASE_PATH}/{path}")
    )

TEST_PATHS = [
    "X_test_con_outliers.xlsx",
    "X_test_sin_outliers.xlsx",
    "X_test_con_outliers_norm.xlsx",
    "X_test_sin_outliers_norm.xlsx",
    "X_test_con_outliers_scal.xlsx",
    "X_test_sin_outliers_scal.xlsx"
]
TEST_DATASETS = []
for path in TEST_PATHS:
    TEST_DATASETS.append(
        pd.read_excel(f"{BASE_PATH}/{path}")
    )

y_train = pd.read_excel(f"{BASE_PATH}/y_train.xlsx")
y_test = pd.read_excel(f"{BASE_PATH}/y_test.xlsx")

results = []
models=[]

for index, dataset in enumerate(TRAIN_DATASETS):
    model = LinearRegression()
    model.fit(dataset, y_train)
    models.append(model)
    
    y_pred_train = model.predict(dataset)
    y_pred_test = model.predict(TEST_DATASETS[index])

    results.append(
        {
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "test_r2": r2_score(y_test, y_pred_test)
        }
    )

best_model = 2
final_model = models[1]

with open("../models/linear_best_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("../models/final_results.json", "w") as f:
    json.dump(results, f, indent=4)