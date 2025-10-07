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

        y_train = df_processed['Score']
        X_train = df_processed.drop(['Score'], axis=1)

        scaler = MinMaxScaler()

        numeric_cols = ['Popularity', 'Favorites', 'Completed', 'Dropped', 'Plan to Watch']
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

        self.config = GlobalPreprocessorConfig(
            expected_columns=X_train.columns.tolist(),
            numeric_cols=numeric_cols,
            scaler=scaler
        )

        return X_train, y_train

    def transform(self, df):
        df_processed = self._process(df, is_training=False)
        X = df_processed.drop(columns=['Score'], errors='ignore')

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
            df.drop(subset=['Score'], inplace=True)

        df = df.drop(labels=['Ranked', 'Score-10', 'Score-9', 'Score-8', 'Score-7', 'Score-6',
                             'Score-5', 'Score-4', 'Score-3', 'Score-2', 'Score-1', 'Ranked',
                             'Name', 'English name', 'Japanese name', 'Aired', 'Producers',
                             'Licensors', 'Duration', 'Members', 'Watching', 'On-Hold', 'MAL_ID'])

        numeric_cols = ['Score', 'Episodes']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handling NaN
        df["Episodes"] = df['Episodes'].fillna(0)
        df.dropna(subset=["Score"], inplace=True)

        # Multi-hot encoding for Genres
        genre_dummies = df['Genres'].str.get_dummies(sep=', ')
        df = pd.concat([df, genre_dummies], axis=1)
        df.drop(columns=['Genres'], inplace=True)

        # Extracting Year from Premier
        df[['Season', 'Year']] = df['Premiered'].str.split(' ', expand=True)
        df.drop(columns=['Season', 'Premiered'], inplace=True)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0)
        df['Year'] = 2025 - df['Year']
        df['Age_Cat'] = pd.cut(
            df['Year'],
            bins=[-1, 2, 5, 10, 20, np.inf],
            labels=['new', 'recent', 'modern', 'old', 'classic']
        )
        df = pd.get_dummies(df, columns=['Age_Cat'], drop_first=True)
        df = df.drop(columns=['Year'])

        # Dividing episodes into length categories
        df['Episode_Cat'] = pd.cut(
            df['Episodes'],
            bins=[0, 1, 10, 18, 26, 57, np.inf],
            labels=['single', 'short', 'one_season', 'two_season', 'long', 'very_long']
            # 1, 2-10, 11-18, 18-26,25-57,58-
        )
        df = pd.get_dummies(df, columns=['Episode_Cat'], drop_first=True)
        df = df.drop(columns=['Episodes'])

        # One-hot encoding for Rating, Type and top Studious
        type_dummies = pd.get_dummies(df['Type'], prefix='Type')
        df = pd.concat([df, type_dummies], axis=1)

        rating_dummies = pd.get_dummies(df['Rating'].str.split(' - ', expand=True)[0], prefix='Rating')
        df = pd.concat([df, rating_dummies], axis=1)

        if 'Studios' in df.columns:
            if is_training:
                top_studios = df['Studios'].value_counts().nlargest(10).index
                self._top_studios = top_studios  # Save for transform
            else:
                # During transform: use saved top studios
                top_studios = getattr(self, '_top_studios', [])

            df['Studios'] = df['Studios'].apply(lambda x: x if x in top_studios else 'Other')
            studio_dummies = pd.get_dummies(df['Studios'], prefix='Studio')
            df = pd.concat([df, studio_dummies], axis=1)

        df.drop(columns=['Type', 'Rating', 'Studios', 'Rating_PG-13', 'Type_TV'], inplace=True)

        # Bool columns to int
        bool_cols = df.select_dtypes('bool').columns
        df[bool_cols] = df[bool_cols].astype(int)

        return df
