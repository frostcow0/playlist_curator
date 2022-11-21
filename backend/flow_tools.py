import os
import logging
from typing import Tuple
from datetime import date
import pandas as pd
import spotipy as spot

from content_based import CosineDistance, MinkowskiDistance, ohe, encode_year


# CONFIG = json.load(
#     open(file="config.json",
#     encoding="utf-8"))
SCOPE = ["user-read-recently-played",
    "playlist-modify-public",
    "user-library-read",]

def set_env_variables():
    """Manually sets environment variables"""
    os.environ["SPOTIPY_CLIENT_ID"] = "468b8b024bfb41d5b1957dad2afc766a"
    os.environ["SPOTIPY_CLIENT_SECRET"] = "8827668f8ed64f13bf8c2e83781c3997"
    os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:8080"

def get_token():
    """Returns Spotify token string from config credentials.
    Uses the Authorization flow.

    :param config (dict): Global config dict
    :param username (str): Username for token
    :return token (str): Spotify authorization token
    """
    return spot.util.prompt_for_user_token(
        scope=SCOPE)

def parse_track_info(item) -> list:
    """Puts together a list of info from a response['item'].

    :param item (dict): Item from API response
    :return info (list): List of track info
    """
    track = item['track']
    track_id = track['id']
    song_name = track['name']
    album_name = track['album']['name']
    artist_name = track['artists'][0]['name']
    explicit = track['explicit']
    song_popularity = track['popularity']
    album_release_date = track['album']['release_date']
    return [track_id, song_name, album_name,
        artist_name, explicit, song_popularity, album_release_date]

def get_tracks_info(client:spot.Spotify, tracks:list) -> pd.DataFrame:
    """Gets info for provided track IDs"""
    response = client.tracks(tracks)
    tracks_info = [[track['name'], track['artists'][0]['name']] for track in response['tracks']]
    tracks_df = pd.DataFrame(tracks_info, columns=['Song Name', 'Artist Name'])
    tracks_df.index = [idx+1 for idx in tracks_df.index]
    return tracks_df

def get_last_50_songs(client:spot.Spotify) -> pd.DataFrame:
    """Using the Spotify client, returns dataframe
    of the user's last 50 played songs.

    :param client (spot.Spotify): Spotify client
    :return df (pd.DataFrame): 50 last played songs
    """
    response = client.current_user_recently_played()
    all_tracks = [parse_track_info(item) for item in response['items']]
    headers = ['Track ID', 'Song Name',
        'Album Name', 'Artist Name', 'Explicit',
        'Song Popularity', 'Album Release Date']
    tracks_df = pd.DataFrame(all_tracks, columns=headers)
    # Replace Album Release Date with Album Release Year
    tracks_df["Album Release Year"] = pd.to_datetime(
        tracks_df["Album Release Date"],
        errors="coerce").dt.year
    tracks_df.drop(["Album Release Date"], axis=1, inplace=True)
    logging.info(" Formatted last 50 played tracks in a DataFrame: \n%s",
        tracks_df.iloc[0])
    return tracks_df

def get_saved_tracks(client:spot.Spotify, limit:int=50) -> pd.DataFrame:
    """Using the Spotify client, returns dataframe
    of the user's last n saved tracks.

    :param client (spot.Spotify): Spotify client
    :param limit (int): Number of songs to retrieve
    """
    headers = ['Track ID', 'Song Name',
        'Album Name', 'Artist Name', 'Explicit',
        'Song Popularity', 'Album Release Date']
    results = []
    if limit > 50: # Spotify's request limit is 50
        counter = 0
        while limit > 0:
            if limit//50:
                limit -= 50
                n = 50
            else:
                n, limit = limit, 0
            response = client.current_user_saved_tracks(limit=n, offset=counter*50)
            tracks = [parse_track_info(item) for item in response['items']]
            results.append(pd.DataFrame(tracks, columns=headers))
            counter += 1
    else:
        response = client.current_user_saved_tracks(limit=limit)
        tracks = [parse_track_info(item) for item in response['items']]
        results.append(pd.DataFrame(tracks, columns=headers))
    tracks_df = pd.concat(results, ignore_index=True)
    # Replace Album Release Date with Album Release Year
    tracks_df["Album Release Year"] = pd.to_datetime(
        tracks_df["Album Release Date"],
        errors="coerce").dt.year
    tracks_df.drop(["Album Release Date"], axis=1, inplace=True)
    logging.info(" Formatted %s most recent saved tracks in a DataFrame: \n%s",
        tracks_df.shape[0], tracks_df.iloc[0])
    # Remove duplicate song name/artist name combos
    # incase the user liked a single & album version of a song
    tracks_df = tracks_df.drop_duplicates(
        subset=["Song Name", "Artist Name"],
        keep="last").reset_index(drop=True)
    return tracks_df

def prep_dataframes(saved:pd.DataFrame, last:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Helper function for get_recommendation"""

    # Not sure what to normalize and what not to...
    # I think it's fine for the distance measures I'm using
    # and probably for Random Forest as well, but past that...? no

    # cols_to_normalize = ['']
    # saved['duration_norm'] = normalize(saved['Duration'].values)
    # saved['song_popularity'] = normalize(saved['Song Popularity'].values)
    # last['duration_norm'] = normalize(last['Duration'].values)
    # last['song_popularity'] = normalize(last['Song Popularity'].values)

    # OHE Explicit, Ordinally Encode the Album Release Year
    saved, last = ohe(saved=saved, last=last)
    saved, last = encode_year(saved=saved, last=last)

    # Drop columns not used in recommendation
    cols = ['Song Popularity', 'Song Name', 'Album Name', 'Artist Name']
    saved.drop(columns = cols, inplace = True)
    last.drop(columns = cols, inplace = True)

    # Set the index of each to the Track ID
    saved.set_index('Track ID', inplace = True)
    last.set_index('Track ID', inplace = True)

    logging.info(" Prepped saved & last played songs")
    return saved, last

def run_similarity(method:str, saved:pd.DataFrame, last:pd.DataFrame, n_rec:int=5) -> pd.DataFrame:
    """Uses cosine similarity to get recommended songs from the user's saved songs"""
    # Prepping data
    prepped_saved, prepped_last = prep_dataframes(saved, last)
    prepped_last.reset_index(drop=True, inplace=True)
    averaged_vector = prepped_last.mean(axis=0)

    # Map of different recommender classes
    recommenders = {
        "cosine": CosineDistance(),
        "minkowski": MinkowskiDistance(p_value=2),
    }

    # Get the class for the method
    method_class = recommenders.get(method)

    # Call the class's recommend method and return the result
    distance = method_class.recommend(goal_vector=averaged_vector,
        comparison_set=prepped_saved, n_rec=n_rec)
    return distance

def create_playlist(client:spot.Spotify, user_id:str) -> str:
    """Creates playlist for the user"""
    today = date.today().strftime("%m/%d/%Y")
    playlist_name = f"{today} Playlist"
    playlist_description = ("This playlist was made by Jon's Playlist Curator"
        f" on {today}. I hope you like it!")
    client.user_playlist_create(user=user_id,
        name=playlist_name, description=playlist_description)
    logging.info(" Created the playlist for the user")
    return playlist_name

def add_playlist_songs(client:spot.Spotify, recommended:pd.DataFrame, user_id:str) -> None:
    """Adds songs to user's most recently made playlist"""
    created_playlist = client.user_playlists(user=user_id, limit=1)
    playlist_id = created_playlist['items'][0]['id']
    client.user_playlist_add_tracks(user=user_id, playlist_id=playlist_id,
        tracks=recommended.index)
    logging.info(" Added the recommended tracks to the playlist")

def add_audio_features(client:spot.Spotify, tracks:pd.DataFrame, limit:int=50) -> pd.Series:
    """Uses's Spotify's audio_features api call to build a Series of
    tracks and their features"""
    results = []
    if limit > 50: # Spotify's request limit is 50
        counter = 0
        while limit > 0:
            if limit//50:
                limit -= 50
                n = 50
            else:
                n, limit = limit, 0
            start = counter * 50
            end = start + 50
            track_ids = tracks.loc[start:end ,"Track ID"]
            result = client.audio_features(track_ids)
            results.append(pd.DataFrame.from_dict(result))
            counter += 1
    else:
        track_ids = tracks.loc[:, "Track ID"]
        result = client.audio_features(track_ids)
        results.append(pd.DataFrame.from_dict(result))
    result_df = pd.concat(results, ignore_index=True)
    df = pd.merge(tracks, result_df,
        left_on="Track ID", right_on="id")
    df.drop(labels=["id", "uri", "track_href",
        "analysis_url", "type"], axis=1, inplace=True)
    # There's a better way to merge using index that
    # doesn't create this 0 column, but I don't remember it :)
    if 0 in df.columns:
        df.drop(labels=[0], axis=1, inplace=True)
    return df

def get_user_data() -> dict:
    """Gets user's saved and recently played songs"""
    # n songs to retrieve from user's saved songs
    saved_count = 2000
    # Get token & Spotify client to get last 50 songs
    set_env_variables() # for running locally
    logging.info(" Requesting token")
    token = get_token()
    logging.info(" Creating Spotify client")
    spotify = spot.Spotify(auth=token)
    logging.info(" Getting saved tracks")
    saved = get_saved_tracks(spotify, limit=saved_count)
    logging.info(" Getting audio features for saved tracks")
    feature_saved = add_audio_features(spotify, saved, limit=saved_count)
    logging.info(" Getting recently played songs")
    last_50 = get_last_50_songs(spotify)
    logging.info(" Getting audio features for recent tracks")
    feature_50 = add_audio_features(spotify, last_50)
    # Has to be in dict format for FastAPI
    return {
        "feature_saved": feature_saved.to_json(),
        "feature_50": feature_50.to_json(),
    }

def get_recommendations(song_data:dict, method:str="cosine") -> tuple:
    """Testing flow"""
    set_env_variables() # for running locally
    logging.info(" Requesting token")
    token = get_token()
    logging.info(" Creating Spotify client")
    spotify = spot.Spotify(auth=token)
    print()
    logging.info(" Getting recommendations using %s distance", method)
    recommended = run_similarity(method=method, saved=song_data["feature_saved"],
        last=song_data["feature_50"], n_rec=20)
    logging.info(" Formatting recommendations")
    nice_format_recommend = get_tracks_info(spotify, recommended.index)
    logging.info(" The recommended songs are: \n%s",
        nice_format_recommend)
    return nice_format_recommend, recommended

def save_playlist(recommended:pd.DataFrame):
    """Saves recommended songs as a playlist for the user"""
    token = get_token()
    spotify = spot.Spotify(auth=token)
    user_id = spotify.current_user()['id']
    create_playlist(spotify, user_id)
    add_playlist_songs(spotify, recommended, user_id)

    # Future additions:
    #   add genre to the songs (from artist)
    #   add more audio features per song (heavy compute cost, big benefit)
    #   store saved songs to add collaborative filtering/hybrid algorithm
    #   edit playlist songs if they've already used this to make a playlist today


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = get_user_data()
    # get_recommendations(data)
    # data['feature_saved'].to_csv('saved.csv')
    # data['feature_50'].to_csv('recent.csv')
