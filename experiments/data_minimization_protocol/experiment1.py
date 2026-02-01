# Baseline for research
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Load Adult dataset
adult = fetch_ucirepo(id=2)

X = adult.data.features
y = adult.data.targets

# 확인
print(type(X))
print(X.head(5))
print(X.columns)
print(X.shape)
print(X.dtypes)
print(y.value_counts())

# EDA

# preprocessing
# 타깃 변수 변환
y = y.iloc[:, 0].apply(lambda x: 1 if '>50K' in x else 0)

# 결측값 처리
X = X.replace('?', pd.NA)
X = X.dropna()
y = y.loc[X.index]

# One-Hot Encoding
X_encoded = pd.get_dummies(X, drop_first=True) # 첫번째 범주를 삭제 후 1로 기준 삼고 나머지를 0으로 만드는 더미 변수가 생성됨
print(X_encoded.shape)

# Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size = 0.2, random_state=42
)

# Model Fitting
# Full Feature Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline_full = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("clf", LogisticRegression(max_iter=2000))
])

pipeline_full.fit(X_train, y_train)

y_pred_full = pipeline_full.predict(X_test)
acc_full = accuracy_score(y_test, y_pred_full)

# 85% : baseline
print(f'Full feature accuracy: {acc_full:.4f}')

# 실험 B - Data Minimization Model
# 준식별자 quasi-identifiers 제거 : age, education, occupation
cols_to_remove = [
    col for col in X_encoded.columns
    if col.startswith('age')
    or col.startswith('education')
    or col.startswith('occupation')
]

X_reduced = X_encoded.drop(columns=cols_to_remove)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

pipeline_reduced = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("clf", LogisticRegression(max_iter=3000, solver="lbfgs"))
])

pipeline_reduced.fit(X_train_r, y_train_r)
y_pred_reduced = pipeline_reduced.predict(X_test_r)

acc_reduced = accuracy_score(y_test_r, y_pred_reduced)

print(f"Data-minimized accuracy: {acc_reduced:.4f}")

