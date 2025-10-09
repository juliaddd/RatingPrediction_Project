from src.pipeline.inference_pipeline import InferencePipeline
from src.data.loader import get_anime
import configparser


def main():
    # Setup
    config = configparser.ConfigParser()
    config.read('config.ini')
    CLIENT_ID = config.get('USER', 'CLIENT_ID').strip()
    pipeline = InferencePipeline()
    pipeline.load_models(
        'models/global_model.joblib',
        'models/user_model_Stark700.joblib'
    )

    # Get anime data
    anime = get_anime('5114', CLIENT_ID)  # FMA:B


    # Predict
    prediction = pipeline.predict(anime)
    print(f"Prediction: {prediction}")


if __name__ == '__main__':
    main()
