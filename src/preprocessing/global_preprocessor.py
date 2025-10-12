import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler


@dataclass
class GlobalPreprocessorConfig:
    expected_columns: List[str]
    numeric_cols: List[str]
    scaler: MinMaxScaler
    top_studios: List[str]
    fill_values: Dict[str, float]


class GlobalPreprocessor:
    def __init__(self):
        self.config = None
        self._top_studios = []
        self._fill_values = {}

    def fit(self, df):

        df_processed = self._process(df, is_training=True)

        y_train = df_processed['score']
        X_train = df_processed.drop(['score'], axis=1)

        scaler = MinMaxScaler()
        numeric_cols = ['popularity', 'favorites', 'completed', 'dropped', 'plan to watch']
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

        self.config = GlobalPreprocessorConfig(
            expected_columns=X_train.columns.tolist(),
            numeric_cols=numeric_cols,
            scaler=scaler,
            top_studios=self._top_studios,
            fill_values=self._fill_values,
        )

        return X_train, y_train

    def transform(self, df):

        df_processed = self._process(df, is_training=False)

        X_test = df_processed.drop(['score'], axis=1)

        if self.config.numeric_cols:
            cols_to_scale = [col for col in self.config.numeric_cols if col in X_test.columns]
            if cols_to_scale:
                X_test[cols_to_scale] = self.config.scaler.transform(X_test[cols_to_scale])

        missing_cols = [col for col in self.config.expected_columns if col not in X_test.columns]

        if missing_cols:
            missing_df = pd.DataFrame(0, index=df_processed.index, columns=missing_cols)
            X_test = pd.concat([df_processed, missing_df], axis=1)

        extra_cols = set(X_test) - set(self.config.expected_columns) - {'score'}
        if extra_cols:
            X_test = X_test.drop(columns=list(extra_cols), errors='ignore')

        X_test = X_test[self.config.expected_columns]
        X_test = X_test.astype('float64')

        return X_test

    def _process(self, df,  is_training=True):
        df = df.copy()

        columns_to_drop =['ranked', 'score-10', 'score-9', 'score-8', 'score-7', 'score-6',
                             'score-5', 'score-4', 'score-3', 'score-2', 'score-1',
                             'name', 'english name', 'japanese name', 'aired', 'producers',
                             'licensors', 'duration', 'members', 'watching', 'on-hold', 'mal_id', 'source']

        existing_cols = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_cols)

        df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df = df.dropna(subset=['score'])

        # Handling NaN
        num_cols = ['popularity', 'favorites', 'completed', 'dropped', 'plan to watch', 'episodes']
        if is_training:
            df = df.dropna(subset=['score'])
        if is_training:
            self._fill_values = {}
            for col in num_cols:
                if col in df.columns:
                    if col == 'episodes':
                        self._fill_values[col] = 12
                    else:
                        self._fill_values[col] = df[col].median()
        for col in num_cols:
            if col in df.columns:
                fill_value = self._fill_values.get(col, 0)
                df[col] = df[col].fillna(fill_value)

        categorical_cols = ['genres', 'type', 'rating',  'studios']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                df[col] = df[col].replace('', 'Unknown')
                df[col] = df[col].astype(str)

        # Multi-hot encoding for Genres
        genre_dummies = (
            df['genres']
            .str.split(', ', expand=True)
            .stack()
            .str.get_dummies()
            .groupby(level=0)
            .sum()
            .add_prefix('Genre_')
        )
        df = pd.concat([df, genre_dummies], axis=1)
        df.drop(columns=['genres'], inplace=True)

        current_year = datetime.now().year
        # Extracting Year from Premier
        if 'premiered' in df.columns:
            df['premiered'] = df['premiered'].fillna(f'Unknown {current_year}')
            df['premiered'] = df['premiered'].replace('', f'Unknown {current_year}')
            df['premiered'] = df['premiered'].astype(str)

            split_result = df['premiered'].str.split(' ', expand=True)

            if split_result.shape[1] == 1:
                split_result[1] = current_year

            df['season'] = split_result[0]
            df['year'] = split_result[1]

            df.drop(columns=['season', 'premiered'], inplace=True)
            df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0)
            df['year'] = current_year - df['year']
            df['age_cat'] = pd.cut(
                df['year'],
                bins=[-1, 2, 5, 10, 20, np.inf],
                labels=['new', 'recent', 'modern', 'old', 'classic']
            )
            df = pd.get_dummies(df, columns=['age_cat'], drop_first=True)
            df = df.drop(columns=['year'])

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
                self._top_studios = top_studios
            else:
                # During transform: use saved top studios
                top_studios = self._top_studios

            df['studios'] = df['studios'].apply(lambda x: x if x in top_studios else 'Other')
            studio_dummies = pd.get_dummies(df['studios'], prefix='Studio')
            df = pd.concat([df, studio_dummies], axis=1)

        df = df.drop(columns=['type', 'rating', 'studios', 'rating_PG-13', 'type_TV'], errors='ignore')

        # Bool columns to int
        bool_cols = df.select_dtypes('bool').columns
        df[bool_cols] = df[bool_cols].astype(int)
        return df
