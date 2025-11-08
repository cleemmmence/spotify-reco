import pandas as pd
url = "https://raw.githubusercontent.com/cleemmmence/spotify-reco/refs/heads/main/spotify_songs.csv"
dsf = pd.read_csv(url)
print(dsf.head(5))