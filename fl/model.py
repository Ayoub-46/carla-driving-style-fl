import xgboost as xgb
from sklearn.metrics import accuracy_score

class DrivingStyleModel:
    def __init__(self):
        self.params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'eta': 0.1,
            'max_depth': 6,
            'eval_metric': 'merror',
            'nthread': 4
        }
        self.bst = None

    def train(self, X_train, y_train, num_boost_round=1, xgb_model=None):
        """
        Train the model. 
        Note: xgb_model can be a path or a Booster object to continue training.
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.bst = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            xgb_model=xgb_model
        )
        return self.bst

    def evaluate(self, X_test, y_test):
        dtest = xgb.DMatrix(X_test, label=y_test)
        preds = self.bst.predict(dtest)
        acc = accuracy_score(y_test, preds)
        return acc