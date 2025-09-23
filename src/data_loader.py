#Loading data from user API
import pandas as pd
import requests
def get_user_animelist(username: str, client_id: str):
    # Loads list of anime from MAL for a given user
    url = f'https://api.myanimelist.net/v2/users/{username}/animelist?limit=500'
    headers = {
        'X-MAL-CLIENT-ID': client_id
    }
    params = {
        'fields': 'id, title , my_list_status, start_season{year}, mean, genres, popularity, studios',
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
        anime = item['node']
        score = item.get('my_list_status', {}).get('score', None)
        rows.append({
            "id": anime['id'],
            "title": anime['title'],
            "mean": anime.get('mean'),
            "genres": [g['name'] for g in anime.get('genres', [])],
            "popularity": anime.get('popularity'),
            "score": score,
            "studios": [s['id'] for s in anime.get('studios', [])],
        })

    return pd.DataFrame(rows)

