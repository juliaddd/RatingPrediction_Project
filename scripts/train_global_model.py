import configparser
from sklearn.model_selection import train_test_split

from src.data.loader import load_kaggle_data
from src.models.global_model import GlobalModel


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    data_path = config.get('DATA', 'KAGGLE_PATH').strip()

    print(f"\nLoading data from {data_path}...")
    try:
        df = load_kaggle_data(data_path)
        print(f"Loaded {len(df)} anime entries")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    df = load_kaggle_data(data_path)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    model = GlobalModel()

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
    model_path = output_dir + f"global_model.joblib"
    try:
        model.save(model_path)
    except Exception as e:
        print(f"Error saving the model: {e}")
        return


if __name__ == '__main__':
    main()
