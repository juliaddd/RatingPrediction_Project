import streamlit as st
import sys
import configparser
from pathlib import Path
from src.data.loader import search_anime, get_anime
from src.pipeline.inference_pipeline import InferencePipeline
sys.path.append('.')


st.set_page_config(
    page_title="Anime rating prediction",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'mailto:dobregojulia@gmail.com',
        'Report a bug': "mailto:dobregojulia@gmail.com",
        'About': "If you are an anime fan, this app will be useful to predict how you would score anime. "
                 "Statistics for your animelist coming soon!"
    }
)
if st.sidebar.button("Reset All"):
    st.session_state.clear()
    st.rerun()


@st.cache_resource
def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config.get('USER', 'CLIENT_ID').strip()


CLIENT_ID = load_config()


def check_user_model(username: str) -> bool:
    """Check if user model exists."""
    model_path = Path(f"models/user_model_{username}.joblib")
    return model_path.exists()


@st.cache_resource
def load_pipeline(username: str) -> InferencePipeline:
    """Load pipeline with user model."""
    pipeline = InferencePipeline()

    global_path = "models/global_model.joblib"
    user_path = f"models/user_model_{username}.joblib"

    if not Path(global_path).exists():
        st.error(f"Global model not found at {global_path}")
        st.info("Run: python -m scripts.train_global_model")
        st.stop()

    if not Path(user_path).exists():
        st.error(f"User model not found for '{username}'")
        st.info(f"Run: python -m scripts.train_personal_model username {username}")
        st.stop()

    pipeline.load_models(global_path, user_path)
    return pipeline


if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'search_results' not in st.session_state:
    st.session_state['search_results'] = None
if 'selected_anime' not in st.session_state:
    st.session_state['selected_anime'] = None
if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None
if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = False

st.title("Anime prediction")
st.subheader("Step 1. Enter your username on MyAnimeList")
username_input = st.text_input(
    "MyAnimeList Username",
    placeholder="Your MAL username...",
    help="Username from myanimelist.net"
)

if username_input and username_input != st.session_state.username:
    st.session_state.username = username_input
    with st.spinner(text="Loading models..."):
        try:
            pipeline = load_pipeline(st.session_state.username)
            st.session_state.pipeline = pipeline
            st.success(f"Model loaded for {username_input}!")
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.pipeline = None

if not st.session_state.pipeline:
    st.info("Enter your username to continue")
    st.stop()


st.subheader("Step 2. Enter anime title")

col1, col2 = st.columns([0.7, 0.3], gap='small', vertical_alignment='bottom')

with col1:
    anime_title = st.text_input(
        "Anime title",
        placeholder="Enter anime title...",
        help="Enter anime for which you want to predict score"
    )
with col2:
    search_button = st.button("Search", type="primary", use_container_width=True)

if anime_title and search_button:
    with st.spinner(f"Searching for anime {anime_title}..."):
        try:
            result = search_anime(anime_title, CLIENT_ID)
            if result:
                st.session_state.search_results = result
            else:
                st.warning("No results found. Try a different title.")
                st.session_state.search_results = None
        except Exception as e:
            st.error(f"Error: {e}")

if not st.session_state.search_results:
    st.info("Enter a valid anime title first")
    st.stop()

st.subheader("Step 3. Select anime for prediction")
results = st.session_state.search_results
for idx, anime in enumerate(results):
    with st.container():
        col1, col2 = st.columns([3, 2])
        with col1:
            if anime.get('img'):
                st.image(anime['img'], width=100)
            else:
                st.text("No Image")

        with col2:
            if anime.get('title'):
                st.text(anime['title'])
            else:
                st.text("No title")

anime_titles = [anime['title'] for anime in results]
selected_anime = st.selectbox(label="Select one of search results", options=anime_titles)
selected_anime = next((anime for anime in results if anime['title'] == selected_anime), None)
if selected_anime:
    if st.button(f"Predict rating", type="primary", use_container_width=True):
        with st.spinner("Predicting..."):
            try:
                anime_data = get_anime(selected_anime['id'], CLIENT_ID)
                prediction = st.session_state.pipeline.predict(anime_data)
                st.session_state.last_prediction = {
                    'anime_data': anime_data,
                    'prediction': float(prediction)
                }
                st.session_state.show_prediction = True

                st.rerun()

            except Exception as e:
                st.error(f"Prediction error: {e}")
                import traceback

                st.code(traceback.format_exc())


if st.session_state.show_prediction and st.session_state.last_prediction:
    anime_data = st.session_state.last_prediction['anime_data']
    prediction = st.session_state.last_prediction['prediction']

    st.subheader(f"{anime_data['title']}")

    col1, col2 = st.columns(2)

    with col1:
        st.text(f"MAL Score: {anime_data.get('mean', 'N/A')}/10")
        st.text(f"Year: {anime_data.get('year', 'Unknown')}")

    with col2:
        st.text(f"Genres: {anime_data.get('genres', 'Unknown')}")
        st.text(f"Type: {anime_data.get('type', 'Unknown')}")

    st.subheader("Prediction")

    if prediction >= 8.5:
        color = "green"
        emoji = "🌟"
        rec = "HIGHLY RECOMMENDED"
    elif prediction >= 7.5:
        color = "blue"
        emoji = "👍"
        rec = "RECOMMENDED"
    elif prediction >= 6.5:
        color = "orange"
        emoji = "🤔"
        rec = "MAYBE"
    else:
        color = "red"
        emoji = "❌"
        rec = "NOT RECOMMENDED"

    st.markdown(f"""
        <div style='
            background-color: rgba(0,0,0,0.05);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid {color};
            text-align: center;
        '>
            <h1 style='margin: 0; color: {color};'>{emoji} {prediction:.1f}/10</h1>
            <p style='margin: 10px 0 0 0; color: gray;'>{rec}</p>
        </div>
        <br>
    """, unsafe_allow_html=True)

with st.expander("Show Detailed Explanation", expanded=False):
    if not st.session_state.show_prediction:
        st.error('Predict first')
    else:
        with st.spinner("Preparing explanation..."):
            try:
                explanation = st.session_state.pipeline.explain(st.session_state.last_prediction['anime_data'])

                st.subheader("PREDICTION BREAKDOWN")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Components**")
                    st.text(f"Global:   {explanation['components']['global_prediction']:.2f}")
                    st.text(f"Personal: {explanation['components']['personal_prediction']:.2f}")
                    st.text(f"MAL Mean: {explanation['components']['site_mean']:.2f}")

                with col2:
                    st.markdown(f"**Weights** ")
                    st.text(f"Global:   {explanation['weights']['global']:.1%}")
                    st.text(f"Personal: {explanation['weights']['personal']:.1%}")
                    st.text(f"Site:     {explanation['weights']['site']:.1%}")

                with col3:
                    st.markdown("**Contributions**")
                    st.text(f"Global:   +{explanation['contributions']['global']:.2f}")
                    st.text(f"Personal: +{explanation['contributions']['personal']:.2f}")
                    st.text(f"Site:     +{explanation['contributions']['site']:.2f}")

                st.caption(f" weights are based on {explanation['num_scoring']:,} scores")

            except Exception as e:
                st.error(f"❌ Explanation error: {e}")
