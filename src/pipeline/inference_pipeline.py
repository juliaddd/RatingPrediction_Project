import pandas as pd
from src.models.global_model import GlobalModel
from src.models.user_model import UserModel
from src.pipeline.ensemble import get_weights


class InferencePipeline:
    def __init__(self):
        self.global_model = GlobalModel()
        self.user_model = UserModel()
        self._models_loaded = False

    def load_model(self, global_path, personal_path):
        global_model = GlobalModel.load(global_path)
        personal_model = UserModel.load(personal_path)

        self.global_model = global_model
        self.user_model = personal_model
        self._models_loaded = True

    def predict(self, anime_data: dict):

        if not self._models_loaded:
            raise ValueError("Models not loaded! Call load_models() first")

        site_mean = anime_data.get('mean', 7.0)
        num_scoring = anime_data.get('num_scoring_users', 0)

        global_df = self._prepare_for_global(anime_data)
        personal_df = self._prepare_for_personal(anime_data)

        global_pred = self.global_model.predict(global_df)[0]
        personal_pred = self.user_model.predict(personal_df)[0]

        weights = get_weights(num_scoring)
        final_pred = (
                weights['wg'] * global_pred +
                weights['wp'] * personal_pred +
                weights['ws'] * site_mean
        )

        anime_pred = {
            'anime_title': anime_data.get('title', 'Unknown'),
            'final_prediction': round(float(final_pred), 2),
            'components': {
                'global_prediction': round(global_pred, 2),
                'personal_prediction': round(personal_pred, 2),
                'site_mean': round(site_mean, 2)
            },
            'weights': {
                'global': round(weights['wg'], 3),
                'personal': round(weights['wp'], 3),
                'site': round(weights['ws'], 3)
            },
            'contributions': {
                'global': round(weights['wg'] * global_pred, 2),
                'personal': round(weights['wp'] * personal_pred, 2),
                'site': round(weights['ws'] * site_mean, 2)
            },
        }
        return anime_pred

    def _prepare_for_global(self, anime_data: dict):

        mal_rating = anime_data.get('rating', 'pg_13')

        rating_mapping = {
            'g': 'G - All Ages',
            'pg': 'PG - Children',
            'pg_13': 'PG-13 - Teens 13 or older',
            'r': 'R - 17+ (violence & profanity)',
            'r_plus': 'R+ - Mild Nudity',
            'rx': 'Rx - Hentai'
        }
        rating = rating_mapping.get(mal_rating, 'PG-13 - Teens 13 or older')

        for_global = {
            'score': 0,  # Dummy value
            'episodes': anime_data.get('num_episodes', 12),
            'genres': anime_data.get('genres', 'Unknown'),
            'premiered': f"Unknown {anime_data.get('year', 2020)}",
            'type': anime_data.get('type', 'TV'),
            'rating': rating,
            'studios': anime_data.get('studios', 'Unknown'),
            'popularity': anime_data.get('popularity', 1000),
            'favorites': anime_data.get('favorites', 0),
            'completed': anime_data.get('completed', 0),
            'dropped': anime_data.get('dropped', 0),
            'plan to watch': anime_data.get('plan_to_watch', 0)
        }

        df_global = pd.DataFrame([for_global])
        return df_global

    def _prepare_for_personal(self, anime_data: dict):
        return pd.DataFrame([anime_data])