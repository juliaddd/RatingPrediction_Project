# Anime Rating Prediction

A machine learning system that predicts personalized anime ratings by combining global patterns, your personal preferences, and community ratings.

## Features

- **Personalized Predictions**: Trained on your MyAnimeList history
- **Ensemble Model**: Combines three prediction sources with dynamic weights
- **Interactive CLI**: Search and predict ratings for any anime
- **Streamlit GUI**: Search and predict ratings for any anime using pleasing interface
- **Smart Weighting**: Automatically adjusts based on anime popularity
- **Detailed Explanations**: See how predictions are calculated

## Future features
- **Statistics dashboard**: Different charts and plots displaying user watching habits based on user's anime list
- **Anime Recommendations**: Get recommendations for similar anime based on your preferences

### Ensemble Logic

Prediction weights adapt based on anime popularity:

| Anime Type | Personal | Global | Site Mean |
|-----------|----------|--------|-----------|
| Obscure (<500 ratings) | 50% | 40% | 10% |
| Medium (500-5k) | 40% | 35% | 25% |
| Popular (5k-20k) | 30% | 25% | 45% |
| Very Popular (>20k) | 25% | 15% | 60% |

## Installation

### Prerequisites

- Python 3.9+
- MyAnimeList API Client ID ([Get one here](https://myanimelist.net/apiconfig))

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/RatingPrediction_Project.git
cd RatingPrediction_Project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API credentials**

Create `config.ini` in the project root:
```ini
[USER]
CLIENT_ID = your_mal_client_id_here
```

## Usage

### 1. Train Global Model

Train the global model on all anime data (uses `data/anime.csv`):
```bash
python -m scripts.train_global_model
```

### 2. Train Personal Model

Train your personalized model:
```bash
python -m scripts.train_personal_model YourUsername
```
Output: `Creates models/user_model_YourUsername.joblib`

**Options:**
- `--verbose`: Show detailed training metrics

### 3. Predict Ratings using GUI

Run following command:
```bash
streamlit run app/streamlit_app.py
```
You can now view Streamlit app in your browser.

**Requires:**
- `streamlit package installed`
- `models/global_model.joblib (from step 1)`
- `models/user_model_YourUsername.joblib (from step 2)`

### 4. Predict Ratings using CLI

Interactive prediction interface:
```bash
python -m scripts.predict YourUsername
```
**Requires:**
- `models/global_model.joblib (from step 1)`
- `models/user_model_YourUsername.joblib (from step 2)`

**Options:**
- `--explain`: Show detailed prediction breakdown

## Project Structure

```
RatingPrediction_Project/
├── src/
│   ├── data/
│   │   └── loader.py              # MAL API interactions
│   ├── preprocessing/
│   │   ├── preprocessor.py        # Personal model preprocessing
│   │   └── global_preprocessor.py # Global model preprocessing
│   ├── models/
│   │   ├── user_model.py          # Personal model
│   │   └── global_model.py        # Global model
│   └── pipeline/
│       ├── ensemble.py            # Weight calculation
│       └── inference_pipeline.py  # Main prediction pipeline
├── scripts/
│   ├── train_global_model.py      # Train global model
│   ├── train_personal_model.py    # Train personal model
│   └── predict.py                 # Interactive CLI
├── app/
│   └── streamlit_app.py           # Interactive GUI
├── models/                        # Saved models
├── data/                          # Training data
├── config.ini                     # API credentials
└── requirements.txt
```
