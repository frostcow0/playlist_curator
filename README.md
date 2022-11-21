# Song Recommender
A project utilizing Spotify's API and the matching Python library spotipy to create personalized 20-song playlists based on the current user's recent listening history.

## The Process
1. Getting data from an API (Spotify)
2. Data wrangling
3. Prototype model
4. Solidify flow for steps 1-3
5. Deploy using Streamlit
6. Refactor / Best Practices
7. Deploy using FastAPI/Plotly
8. Future plans

## Getting data from an API
It took several reads through [spotipy's docs](https://spotipy.readthedocs.io/en/2.19.0/) to understand the usage of scope and credentials. Once I established a working "client" I was able to test run some API calls in a Jupyter Notebook. The coolest call I encountered was `sp.audio_features` for sure. It returns a variety of Spotify metrics from danceability to liveliness and energy. Thanks to the spotipy Python library this step was relatively short, but it also made me realize that I need to take the time separately to practice making my own API calls.


## Data wrangling
In the same aforementioned Jupyter Notebook I practiced a variety of pandas methods for reshaping the data, reshaping it again, reshaping it some more. My favorite method that I've recently picked up is `pd.pivot_table`, turns out they're not just for Excel!

## Prototype model
For now I've used a Content Based Recommender that uses cosine similarity as the underlying algorithm. This is actually extremely exciting, having this cosine similarity, because it allows me to get a working recommender up and off of the ground. With the potential to store users' liked songs, we can accumulate a larger selection of songs to choose recommendations from. This would allow me to branch the recommender system into a hybrid approach combining content based & collaborative filtering.

## Solidify flow for steps 1-3
I've retroactively cleaned up my code, added comments and logs, and changed the structure here and there for improved adaptability and overall readability. Satisfied with the way that things look & run, I'm moving on to the Flask deployment. One of the hardest parts about this is convincing myself that I shouldn't just hunker down on the recommender and improve it until it's perfect - I also have the overall product to think about. What's a recommender good for if no one can use it?

## Deploy using Streamlit
Unfortunately, while Streamlit cloud is very nice (and free), it doesn't allow for redirects, so there's no way I can currently use that with Spotify's API. Locally though, Streamlit makes for great rapid-prototyping and serves to help me validate changes without printing everything to console.

## Refactor/Best Practices
A project is never truly finished, but there are definitely improvements. Since deploying with Streamlit on the cosine distance recommender, I've since added minkowski distance (and refactored the way these classes are set up to make future additions of methods much easier), ordinally encoded the years, and improved the way I make my API calls.

## Deploy using FastAPI & Plotly
I've just finished setting up my FastAPI implementation. It's definitely easy to stand up and get running, but understanding the pydantic models took some work.