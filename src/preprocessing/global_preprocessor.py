import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler


@dataclass
class GlobalPreprocessorConfig:
    expected_columns: List[str]
    numeric_cols: List[str]
    scaler: MinMaxScaler


class GlobalPreprocessor:
    def __init__(self):
        self.config = None

    def fit(self, df):
        df_processed = self._process(df, is_training=True)

        y_train = df_processed['score']
        X_train = df_processed.drop(['score'], axis=1)

        scaler = MinMaxScaler()

        numeric_cols = ['popularity', 'favorites', 'completed', 'dropped', 'plan to Watch']
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

        self.config = GlobalPreprocessorConfig(
            expected_columns=X_train.columns.tolist(),
            numeric_cols=numeric_cols,
            scaler=scaler
        )

        return X_train, y_train

    def transform(self, df):
        df_processed = self._process(df, is_training=False)
        X = df_processed.drop(columns=['score'], errors='ignore')

        if self.config.numeric_cols:
            cols_to_scale = [col for col in self.config.numeric_cols if col in X.columns]
            if cols_to_scale:
                X[cols_to_scale] = self.config.scaler.transform(X[cols_to_scale])

        #  Align columns

        for col in self.config.expected_columns:
            if col not in X.columns:
                X[col] = 0
        extra_cols = set(X) - set(self.config.expected_columns)
        if extra_cols:
            X = X.drop(columns=list(extra_cols), errors='ignore')

        X = X[self.config.expected_columns]

        return X

    def _process(self, df,  is_training=True):
        df = df.copy()
        if is_training:
            df.drop(subset=['score'], inplace=True)

        df = df.drop(labels=['ranked', 'score-10', 'score-9', 'score-8', 'score-7', 'score-6',
                             'score-5', 'score-4', 'score-3', 'score-2', 'score-1',
                             'name', 'english name', 'japanese name', 'aired', 'producers',
                             'licensors', 'duration', 'members', 'watching', 'on-hold', 'mal_id'])

        numeric_cols = ['score', 'episodes']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handling NaN
        df["episodes"] = df['episodes'].fillna(0)
        df.dropna(subset=["score"], inplace=True)

        # Multi-hot encoding for Genres
        genre_dummies = df['genres'].str.get_dummies(sep=', ')
        df = pd.concat([df, genre_dummies], axis=1)
        df.drop(columns=['genres'], inplace=True)

        # Extracting Year from Premier
        df[['season', 'Year']] = df['premiered'].str.split(' ', expand=True)
        df.drop(columns=['season', 'premiered'], inplace=True)
        df['season'] = pd.to_numeric(df['season'], errors='coerce').fillna(0)
        df['season'] = 2025 - df['season']
        df['age_cat'] = pd.cut(
            df['season'],
            bins=[-1, 2, 5, 10, 20, np.inf],
            labels=['new', 'recent', 'modern', 'old', 'classic']
        )
        df = pd.get_dummies(df, columns=['age_cat'], drop_first=True)
        df = df.drop(columns=['season'])

        # Dividing episodes into length categories
        df['episode_cat'] = pd.cut(
            df['episodes'],
            bins=[0, 1, 10, 18, 26, 57, np.inf],
            labels=['single', 'short', 'one_season', 'two_season', 'long', 'very_long']
            # 1, 2-10, 11-18, 18-26,25-57,58-
        )
        df = pd.get_dummies(df, columns=['episode_cat'], drop_first=True)
        df = df.drop(columns=['episodes'])

        # One-hot encoding for Rating, Type and top Studious
        type_dummies = pd.get_dummies(df['type'], prefix='type')
        df = pd.concat([df, type_dummies], axis=1)

        rating_dummies = pd.get_dummies(df['rating'].str.split(' - ', expand=True)[0], prefix='rating')
        df = pd.concat([df, rating_dummies], axis=1)

        if 'studios' in df.columns:
            if is_training:
                top_studios = df['studios'].value_counts().nlargest(10).index
                self._top_studios = top_studios  # Save for transform
            else:
                # During transform: use saved top studios
                top_studios = getattr(self, '_top_studios', [])

            df['studios'] = df['studios'].apply(lambda x: x if x in top_studios else 'Other')
            studio_dummies = pd.get_dummies(df['studios'], prefix='Studio')
            df = pd.concat([df, studio_dummies], axis=1)

        df.drop(columns=['type', 'rating', 'studios', 'rating_PG-13', 'type_TV'], inplace=True)

        # Bool columns to int
        bool_cols = df.select_dtypes('bool').columns
        df[bool_cols] = df[bool_cols].astype(int)

        return df
