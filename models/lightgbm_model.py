import lightgbm as lgb
from sklearn.metrics import classification_report

def train_lightgbm_binary(X_train, y_train, X_test, y_test):
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)
    print("[LightGBM - Binary]")
    print(classification_report(y_test, model.predict(X_test)))

def train_lightgbm_multiclass(X_train, y_train, X_test, y_test, num_class=4):
    model = lgb.LGBMClassifier(objective='multiclass', num_class=num_class)
    model.fit(X_train, y_train)
    print("[LightGBM - Multi-class]")
    print(classification_report(y_test, model.predict(X_test)))
