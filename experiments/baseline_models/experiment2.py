# Extended version for research
from ucimlrepo import fetch_ucirepo
import pandas as pd


# Load Adult dataset
adult = fetch_ucirepo(id=2)

X = adult.data.features
y = adult.data.targets

# 확인
# print(type(X))
# print(X.head(5))
# print(X.columns)
# print(X.shape)
# print(X.dtypes)
print("target distribution_before:",y.value_counts())

# EDA

# preprocessing
# 타깃 변수 변환
y = y.iloc[:, 0].apply(lambda x: 1 if '>50K' in x else 0)

# 결측값 처리
X = X.replace(r"^\s*\?\s*$", pd.NA, regex=True)# 공백이 섞여 있을 수 있어 정규식 처리
X = X.dropna()
y = y.loc[X.index]

print("target distribution_after:",y.value_counts) # 변환 후 target 확인

# One-Hot Encoding
X_encoded = pd.get_dummies(X, drop_first=True) # 첫번째 범주를 삭제 후 1로 기준 삼고 나머지를 0으로 만드는 더미 변수가 생성됨
print("Encoded_shape", X_encoded.shape)

# Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size = 0.2, random_state=42, stratify=y
) # class 비율 고정
#----------------------------데이터 셋 고정-------------------------------------------#
# 확장3
# Model Fitting
# Full Feature Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# features_by_level function
def remove_features_by_level(X, level):
    '''
    Level 0: Full features
    Level 1: age 제거
    Level 2: age + education 제거
    Level 3: age + education + occupation 제거
    '''
    cols = X.columns

    remove_cols = []

    if level >= 1:
        remove_cols += [c for c in cols if c == "age"]

    if level >= 2:
        remove_cols += [c for c in cols if c.startswith("education")]

    if level >= 3:
        remove_cols += [c for c in cols if c.startswith("occupation")]

    return X.drop(columns=remove_cols)

# Logistic Regression pipeline function
def train_and_eval_logistic(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=3000))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)

results = []
for level in range(4):
    X_train_lvl = remove_features_by_level(X_train, level)
    X_test_lvl = remove_features_by_level(X_test, level)

    acc = train_and_eval_logistic(
        X_train_lvl, X_test_lvl, y_train, y_test
    )

    results.append({
        "Level": level,
        "Num_Features": X_train_lvl.shape[1],
        "Accuracy": round(acc,4)
    })

results_df = pd.DataFrame(results)
print(results_df)
