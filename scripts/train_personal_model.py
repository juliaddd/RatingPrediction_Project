import pandas as pd
import requests
import configparser
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
# my imports
from src.data.loader import to_dataframe, get_user_animelist
from src.models.user_model import UserModel


MIN_SCORE = 0
MIN_SAMPLES_FOR_TRAINING = 50


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    USER_NAME = config.get('USER', 'USER_NAME').strip()
    CLIENT_ID = config.get('USER', 'CLIENT_ID').strip()

    print("Loading data from myAnimeList\n")
    try:
        data = get_user_animelist(USER_NAME, CLIENT_ID)
        df = to_dataframe(data)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    df_rated = df[df['score']>MIN_SCORE]
    if len(df_rated) < MIN_SAMPLES_FOR_TRAINING:
        print("Warning: Not enough data to train model")

    df_train, df_test = train_test_split(df_rated, test_size=0.2, random_state=42)
    model = UserModel()

    try:
        model.train(df_train, True)
    except Exception as e:
        print(f"Error training the model: {e}")
        return

    try:
        metrics = model.evaluate(df_test, verbose=True)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return

    output_dir = "models/"
    model_path = output_dir + f"user_model_{USER_NAME}.joblib"
    try:
        model.save(model_path)
    except Exception as e:
        print(f"Error saving the model: {e}")
        return


if __name__ == '__main__':
    main()