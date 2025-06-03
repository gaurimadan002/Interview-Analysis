import xgboost as xgb
from sklearn.metrics import classification_report

def train_xgboost_binary(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    print("[XGBoost - Binary]")
    print(classification_report(y_test, model.predict(X_test)))

def train_xgboost_multiclass(X_train, y_train, X_test, y_test, num_class=4):
    model = xgb.XGBClassifier(objective='multi:softprob', num_class=num_class,
                              use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    print("[XGBoost - Multi-class]")
    print(classification_report(y_test, model.predict(X_test)))
