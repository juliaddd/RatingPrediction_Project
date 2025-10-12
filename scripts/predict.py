import argparse
import sys
from pathlib import Path
from tabulate import tabulate
from src.data.loader import search_anime, get_anime
from src.pipeline.inference_pipeline import InferencePipeline
import configparser
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def check_user_model(username: str) -> bool:
    """Check if user model exists."""
    model_path = Path(f"models/user_model_{username}.joblib")
    return model_path.exists()


def load_pipeline(username: str) -> InferencePipeline:
    """Load pipeline with user model."""
    pipeline = InferencePipeline()

    global_path = "models/global_model.joblib"
    user_path = f"models/user_model_{username}.joblib"

    if not Path(global_path).exists():
        print(f"Global model not found at {global_path}")
        print("Run: python -m scripts.train_global_model")
        sys.exit(1)

    if not Path(user_path).exists():
        print(f"User model not found for '{username}'")
        print(f"Run: python -m scripts.train_personal_model --username {username}")
        sys.exit(1)

    print(f"Loading models...")
    pipeline.load_models(global_path, user_path)
    print(f"Models loaded for user: {username}\n")

    return pipeline


def search(client_id: str) -> dict:
    """Interactive anime search."""
    while True:
        # Get search query
        query = input("Enter anime title (or 'quit' to exit): ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            sys.exit(0)

        if not query:
            print("Please enter a title\n")
            continue

        # Search
        results = search_anime(query, client_id)

        if not results:
            print("No results found. Try another title.\n")
            continue

        # Display results
        table_data = []
        for i, anime in enumerate(results, 1):
            table_data.append([i, anime['id'], anime['title']])

        print(tabulate(table_data,
                       headers=['#', 'ID', 'Title'],
                       tablefmt='grid'))

        # Get user choice
        while True:
            choice = input(f"\nSelect anime [1-{len(results)}] or 's' to search again: ").strip()

            if choice.lower() == 's':
                print()
                break

            try:
                num = int(choice)
                if 1 <= num <= len(results):
                    return results[num - 1]
                else:
                    print(f"Please enter a number between 1 and {len(results)}")
            except ValueError:
                print("Please enter a valid number")


def predict_and_display(pipeline: InferencePipeline, anime_data: dict,
                        client_id: str, explain: bool = False):
    """Get prediction and display results."""
    # Fetch full anime data
    print(f"\nLoading anime data...\n")
    full_anime = get_anime(str(anime_data['id']), client_id)

    # Display header
    print("=" * 70)
    print(f"{full_anime['title']}")
    print("=" * 70)

    if explain:
        # Detailed explanation
        explanation = pipeline.explain(full_anime)

        print(f"\nPREDICTION BREAKDOWN:")
        print(f"\nFinal Prediction: {explanation['final_prediction']:.2f}/10")
        print(f"\nComponents:")
        print(f"\t• Global Model:    {explanation['components']['global_prediction']:.2f}")
        print(f"\t• Personal Model:  {explanation['components']['personal_prediction']:.2f}")
        print(f"\t• Site Mean (MAL): {explanation['components']['site_mean']:.2f}")

        print(f"\nWeights (based on {explanation['num_scoring']:,} scores):")
        print(f"\t• Global:   {explanation['weights']['global']:.1%}")
        print(f"\t• Personal: {explanation['weights']['personal']:.1%}")
        print(f"\t• Site:     {explanation['weights']['site']:.1%}")

        print(f"\nContributions to final score:")
        print(f"\t• Global:   +{explanation['contributions']['global']:.2f}")
        print(f"\t• Personal: +{explanation['contributions']['personal']:.2f}")
        print(f"\t• Site:     +{explanation['contributions']['site']:.2f}")
    else:
        # Simple prediction
        prediction = pipeline.predict(full_anime)

        print(f"\nPredicted Rating: {prediction:.2f}/10")
        print(f"MAL Site Mean: {full_anime['mean']:.2f}/10")

    print("=" * 70)


def main():

    config = configparser.ConfigParser()
    config.read('config.ini')
    CLIENT_ID = config.get('USER', 'CLIENT_ID').strip()

    parser = argparse.ArgumentParser(description="Predict anime ratings",
                                     epilog="""
                                     Examples:
                                         python -m scripts.predict Stark700
                                         python -m scripts.predict Stark700 --explain
                                             """
                                     )

    parser.add_argument("username",  type=str,
                        help="Your MyAnimeList username")

    parser.add_argument('--explain', action='store_true',
                        help='Show prediction breakdown')

    args = parser.parse_args()

    if not check_user_model(args.username):
        print(f"\nNo trained model found for user '{args.username}'")
        print(f"\nTrain your personal model first:")
        print(f"   python -m scripts.train_personal_model {args.username}")
        sys.exit(1)

    pipeline = load_pipeline(args.username)

    while True:
        # Search for anime
        selected_anime = search(CLIENT_ID)
        predict_and_display(pipeline, selected_anime, CLIENT_ID, args.explain)

        again = input("\nPredict another anime? [Y/n]: ").strip().lower()
        if again in ['n', 'no', 'q', 'quit']:
            print("\nThanks for using Anime Rating Predictor!")
            break

        print("\n" + "-" * 70 + "\n")


if __name__ == '__main__':
    main()





