import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import joblib
from src.preprocessing.global_preprocessor import GlobalPreprocessor


class GlobalModel:
    def __init__(self):
        self.model = None
        self.preprocessor = GlobalPreprocessor()
        self.trained = False
        self.metrics = {}

    def train(self, df_train: pd.DataFrame, verbose: bool = False):

        X_train, y_train = self.preprocessor.fit(df_train)

        if verbose:
            print(f"  Data: {X_train.shape[0]} anime, {X_train.shape[1]} features")
            print(f"  Target range: [{y_train.min():.1f}, {y_train.max():.1f}]")

        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        self.trained = True

        y_pred_train = self.model.predict(X_train)
        self.metrics['train_mae'] = mean_absolute_error(y_train, y_pred_train)
        self.metrics['train_r2'] = r2_score(y_train, y_pred_train)

        if verbose:
            print(f"  Train MAE: {self.metrics['train_mae']:.3f}")
            print(f"  Train R²: {self.metrics['train_r2']:.3f}")
            print("="*40)
            print("Training complete")

    def evaluate(self, df_test: pd.DataFrame, verbose: bool = False):
        if not self.trained:
            raise ValueError("Model not trained!")

        X_test = self.preprocessor.transform(df_test)
        y_test = df_test['score']
        y_pred = self.model.predict(X_test)

        metrics = {
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred),
            'test_rmse': np.sqrt(np.mean((y_test - y_pred) ** 2))
        }

        self.metrics.update(metrics)

        if verbose:
            print(f"\nTest Results:")
            print(f"  MAE: {metrics['test_mae']:.3f}")
            print(f"  R²: {metrics['test_r2']:.3f}")
            print(f"  RMSE: {metrics['test_rmse']:.3f}")
            print("=" * 40)
            print("Evaluation complete")

        return metrics

    def predict(self, anime_data):
        if not self.trained:
            raise ValueError("Model not trained!")

        if isinstance(anime_data, dict):
            anime_data = pd.DataFrame([anime_data])

        X = self.preprocessor.transform(anime_data)
        prediction = self.model.predict(X)

        return np.clip(prediction, 0, 10)

    def save(self, path: str):
        if not self.trained:
            raise ValueError("Model not trained!")

        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'metrics': self.metrics
        }, path)

        print(f"Model successfully saved to {path}")

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)

        predictor = cls()
        predictor.model = data['model']
        predictor.preprocessor = data['preprocessor']
        predictor.metrics = data.get('metrics', {})
        predictor.trained = True

        print(f"Model loaded from {path}")
        return predictor
