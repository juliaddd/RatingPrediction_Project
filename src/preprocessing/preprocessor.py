import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class PreprocessorConfig:
    genre_affinity: Dict[str, float]
    expected_columns: List[str]
    rare_genres: List[str]
    rare_studios: List[str]
    overall_mean: float


class AnimePreprocessor:
    def __init__(self, drop_mean=True):
        self.config = None
        self.drop_mean = drop_mean

    def fit(self, df_train):

        df_processed = self._process(df_train, is_training=True)
        genre_affinity = self._calculate_genre_affinity(df_processed)
        rare_genres, rare_studios = self._calculate_rare_columns(df_processed)

        df_processed = self._group_rare_features(
            df_processed,
            rare_genres,
            rare_studios
        )
        df_processed = self._apply_affinity(df_processed, genre_affinity)

        y_train = df_processed['score']
        X_train = df_processed.drop(columns=['score'], errors='ignore')

        self.config = PreprocessorConfig(
            genre_affinity=genre_affinity,
            expected_columns=X_train.columns.tolist(),
            rare_genres=rare_genres,
            rare_studios=rare_studios,
            overall_mean=y_train.mean()
        )

        return X_train, y_train

    def transform(self, df):
        if self.config is None:
            raise ValueError("Must call fit() before transform()")

        df_processed = self._process(df, is_training=False)
        df_processed = self._group_rare_features(df_processed, self.config.rare_genres, self.config.rare_studios)
        df_processed = self._apply_affinity(df_processed, self.config.genre_affinity)

        missing_cols = [col for col in self.config.expected_columns if col not in df_processed.columns]

        if missing_cols:
            missing_df = pd.DataFrame(0, index=df_processed.index, columns=missing_cols)
            df_processed = pd.concat([df_processed, missing_df], axis=1)

        extra_cols = set(df_processed.columns) - set(self.config.expected_columns) - {'score'}
        df_processed = df_processed.drop(columns=list(extra_cols), errors='ignore')

        df = df_processed[self.config.expected_columns]

        return df

    def fit_transform(self, df_train) -> Tuple[pd.DataFrame, pd.Series]:
        return self.fit(df_train)

    def _process(self, df, is_training=True):

        df = df.copy()

        if is_training:
            df = df[df['score'] > 0]

        df.drop(columns=['title', 'id'], inplace=True)

        categorical_cols = ['studios', 'genres', 'rating', 'status', 'type']
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
            df[col] = df[col].replace('', 'Unknown')

        numerical_cols = ['year', 'mean', 'popularity', 'members', 'num_episodes']
        for col in numerical_cols:
            if col in df.columns:
                default_values = {
                    'year': 2020,
                    'mean': 7.0,
                    'popularity': 1000,
                    'members': 10000,
                    'num_episodes': 12
                }
                df[col] = df[col].fillna(default_values.get(col, df[col].median()))

        df = self._encode(df)

        genre_cols = [col for col in df.columns if col.startswith('Genre_')]
        df['num_genres'] = df[genre_cols].sum(axis=1)

        # TODO: commented for testing
        if self.drop_mean and 'mean' in df.columns:
            df = df.drop(columns=['mean'])

        # Log transformations
        if 'popularity' in df.columns:
            df['popularity'] = np.log1p(df['popularity'])

        if 'members' in df.columns:
            df.drop(columns=['members'], inplace=True)

        if 'num_episodes' in df.columns:
            df['num_episodes'] = np.log1p(df['num_episodes'])

        # Create anime_age if year exists
        if 'year' in df.columns:
            df['anime_age'] = 2025 - df['year']
            df = df.drop(columns=['year'])

        if 'anime_age' in df.columns:
            df['age_category'] = pd.cut(
                df['anime_age'],
                bins=[-1, 2, 5, 10, 20, np.inf],
                labels=['new', 'recent', 'modern', 'old', 'classic']
            )
            df = pd.get_dummies(df, columns=['age_category'], drop_first=True)
            df = df.drop(columns=['anime_age'])

        if 'num_episodes' in df.columns:
            df['episode_cat'] = pd.cut(
                df['num_episodes'],
                bins=[0, 1, 10, 18, 26, 57, np.inf],
                labels=['single', 'short', 'one_season', 'two_season', 'long', 'very_long']
                # 1, 2-10, 11-18, 18-26,25-57,58-
            )
            df = pd.get_dummies(df, columns=['episode_cat'], drop_first=True)
            df = df.drop(columns=['num_episodes'])

        return df

    def _encode(self, df):

        status_dummies = pd.get_dummies(df['status'], prefix='Status')
        df = pd.concat([df, status_dummies], axis=1)

        type_dummies = pd.get_dummies(df['type'], prefix='Type')
        df = pd.concat([df, type_dummies], axis=1)

        rating_dummies = pd.get_dummies(df['rating'], prefix='Rating')
        df = pd.concat([df, rating_dummies], axis=1)

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

        studio_dummies = (
            df['studios']
            .str.split(', ', expand=True)
            .stack()
            .str.get_dummies()
            .groupby(level=0)
            .sum()
            .add_prefix('Studios_')
        )
        df = pd.concat([df, studio_dummies], axis=1)

        df.drop(columns=['genres'], inplace=True)
        df.drop(columns=['studios'], inplace=True)
        df = df.drop(columns=['type', 'Type_tv'], errors='ignore')
        df = df.drop(columns=['rating', 'Rating_pg_13'], errors='ignore')
        df = df.drop(columns=['status', 'Status_completed'], errors='ignore')

        return df

    def _calculate_genre_affinity(self, df_train):

        genre_cols = [col for col in df_train.columns if col.startswith('Genre_')]
        genre_affinity = {}

        overall_mean = df_train['score'].mean()

        for genre_col in genre_cols:
            genre_name = genre_col.replace('Genre_', '')
            mask = df_train[genre_col] == 1

            if mask.sum() >= 5:
                genre_affinity[genre_name] = df_train[mask]['score'].mean()
            else:
                genre_affinity[genre_name] = overall_mean

        return genre_affinity

    def _calculate_rare_columns(self, df_train):

        total_anime = len(df_train)

        genre_columns = [col for col in df_train.columns if col.startswith('Genre_')]
        genre_counts = df_train[genre_columns].sum().sort_values(ascending=False)

        min_count = max(5, int(total_anime * 0.01))
        frequent_genres = genre_counts[genre_counts >= min_count].index.tolist()
        rare_genre_columns = [col for col in genre_columns if col not in frequent_genres]

        studio_columns = [col for col in df_train.columns if col.startswith('Studios_')]
        studio_counts = df_train[studio_columns].sum().sort_values(ascending=False)

        min_count = max(10, int(total_anime * 0.01))
        frequent_studios = studio_counts[studio_counts >= min_count].index.tolist()
        rare_studio_columns = [col for col in studio_columns if col not in frequent_studios]

        return rare_genre_columns, rare_studio_columns

    def _group_rare_features(self, df, rare_genres, rare_studios):
        df = df.copy()
        if rare_genres:
            genre_cols_to_check = [col for col in rare_genres if col in df.columns]
            if genre_cols_to_check:
                df['Genre_Other'] = df[genre_cols_to_check].max(axis=1)
                df = df.drop(columns=genre_cols_to_check)

        if rare_studios:
            studio_cols_to_check = [col for col in rare_studios if col in df.columns]
            if studio_cols_to_check:
                df['Studio_Other'] = df[studio_cols_to_check].max(axis=1)
                df = df.drop(columns=studio_cols_to_check)

        return df

    def _apply_affinity(self, df, genre_affinity):

        genre_cols = [col for col in df.columns if col.startswith('Genre_')]
        for genre_col in genre_cols:
            genre_name = genre_col.replace('Genre_', '')
            if genre_name in genre_affinity:
                affinity_score = genre_affinity[genre_name]
                df[f'affinity_{genre_name}'] = df[genre_col] * affinity_score

        df = df.drop(columns=genre_cols)
        return df
