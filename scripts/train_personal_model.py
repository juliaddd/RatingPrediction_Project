import configparser
import argparse
import sys
from sklearn.model_selection import train_test_split
# my imports
from src.data.loader import to_dataframe, get_user_animelist
from src.models.user_model import UserModel


MIN_SCORE = 0
MIN_SAMPLES_FOR_TRAINING = 50


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    CLIENT_ID = config.get('USER', 'CLIENT_ID').strip()

    parser = argparse.ArgumentParser(description="Train personal model",
                                     epilog="""
                                         Examples:
                                             python -m scripts.train_personal_model Stark700
                                             python -m scripts.train_personal_model Stark700 --verbose
                                                 """
                                     )

    parser.add_argument("username", type=str,
                        help="Your MyAnimeList username")
    parser.add_argument('--verbose', action='store_true',
                        help='Show training breakdown')

    args = parser.parse_args()
    username = args.username.strip()

    if not username:
        print(f"Username cannot be empty")
        sys.exit(1)

    print("Loading data from myAnimeList\n")
    try:
        data = get_user_animelist(username, CLIENT_ID)
        df = to_dataframe(data)
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"\nPlease check:")
        print(f"   • Username '{username}' exists on MyAnimeList")
        print(f"   • Your anime list is public")
        print(f"   • CLIENT_ID in config.ini is valid")
        sys.exit(1)

    df_rated = df[df['score']>MIN_SCORE]
    if len(df_rated) < MIN_SAMPLES_FOR_TRAINING:
        print("Warning: Not enough data to train model")
        print(f"Minimum recommended: {MIN_SAMPLES_FOR_TRAINING}")
        print(f"Model accuracy may be low with limited data.")
        proceed = input(f"\n   Continue anyway? [y/N]: ").strip().lower()
        if proceed not in ['y', 'yes']:
            print("   Training cancelled.")
            sys.exit(0)

    df_train, df_test = train_test_split(df_rated, test_size=0.2, random_state=42)
    model = UserModel()

    try:
        model.train(df_train, args.verbose)
    except Exception as e:
        print(f"Error training the model: {e}")
        sys.exit(1)

    try:
        metrics = model.evaluate(df_test, args.verbose)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)

    output_dir = "models/"
    model_path = output_dir + f"user_model_{username}.joblib"
    try:
        model.save(model_path)
    except Exception as e:
        print(f"Error saving the model: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()