import pandas as pd
import requests


def load_kaggle_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()
    return df


def get_user_animelist(username: str, client_id: str):
    # Loads list of anime from MAL for a given user
    url = f'https://api.myanimelist.net/v2/users/{username}/animelist?limit=500'
    headers = {
        'X-MAL-CLIENT-ID': client_id
    }
    params = {
        'fields': 'id, title, list_status{score,status}, start_season{year}, mean,\
         genres, popularity, media_type, rating, num_episodes, studios, num_list_users,favorites'
    }

    all_data = []
    next_page = url

    while next_page:
        response = requests.get(next_page, headers=headers, params=params if next_page == url else None)
        if response.status_code != 200:
            raise ValueError(f"Error with API request: {response.status_code} - {response.text}")

        data = response.json()
        all_data.extend(data['data'])
        next_page = data.get("paging", {}).get("next")

        print(f"Loaded {len(all_data)} anime...")

    return all_data


def to_dataframe(all_data):
    rows = []
    for item in all_data:
        anime = item.get('node', {})
        score = item.get('list_status', {}).get('score')
        status = item.get('list_status', {}).get('status')
        # num_episodes_watched = item.get('list_status', {}).get('num_episodes_watched')
        year = item.get('node', {}).get('start_season', {}).get('year')
        rows.append({
            "id": anime['id'],
            "title": anime['title'],
            "mean": anime.get('mean'),
            "genres": [g['name'] for g in anime.get('genres', [])],
            "studios": [s['name'] for s in anime.get('studios', [])],
            "rating": anime.get('rating'),
            "year": year,
            "type": anime.get('media_type'),
            "popularity": anime.get('popularity'),
            "score": score,
            "status": status,
            "members": anime['num_list_users'],
            "num_episodes": anime['num_episodes'],
        })

    df = pd.DataFrame(rows)
    df['studios'] = df['studios'].apply(
        lambda x: ', '.join(x) if x else 'Unknown'
    )
    df['genres'] = df['genres'].apply(
        lambda x: ', '.join(x) if x else 'Unknown'
    )

    return df


def search_anime(q: str,  client_id: str):

    url = f"https://api.myanimelist.net/v2/anime"
    headers = {
        'X-MAL-CLIENT-ID': client_id
    }
    params = {
        'q': q,
        'limit': 5,
        'fields': 'id,title'
    }

    response = requests.get(url=url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(f"Error with API request: {response.status_code} - {response.text}")

    data = response.json()

    rows = []
    for item in data['data']:
        anime = item.get('node', {})
        rows.append({
            "id": anime.get('id'),
            "title": anime.get('title')
        })

    return rows


def get_anime(anime_id: str, client_id: str):

    url = f"https://api.myanimelist.net/v2/anime/{anime_id}"
    headers = {
        'X-MAL-CLIENT-ID': client_id
    }
    params = {
        'fields': 'id,title,mean,genres,studios,start_season,media_type,'
                  'rating,num_episodes,popularity,num_list_users'
    }

    response = requests.get(headers=headers, params=params, url=url)

    if response.status_code != 200:
        raise ValueError(f"Error with API request: {response.status_code} - {response.text}")

    data = response.json()
    genres = [g['name'] for g in data.get('genres', [])]
    studios = [s['name'] for s in data.get('studios', [])]

    anime_data = {
        'id': data['id'],
        'title': data['title'],
        'mean': data.get('mean'),
        'genres': ', '.join(genres) if genres else 'Unknown',
        'studios': ', '.join(studios) if studios else 'Unknown',
        'year': data.get('start_season', {}).get('year'),
        'type': data.get('media_type'),
        'rating': data.get('rating'),
        'num_episodes': data.get('num_episodes'),
        'popularity': data.get('popularity'),
        'members': data.get('num_list_users'),
        'score': 0,
        'status': 'plan_to_watch'  # Default
    }

    return anime_data
