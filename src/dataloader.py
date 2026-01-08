import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


continuous_features = ["danceability","energy","loudness","speechiness","acousticness","instrumentalness","liveness",
"valence", "tempo", "duration_mn", "song_age"]

binary_features = ["genre_edm","genre_latin","genre_pop","genre_r&b","genre_rap","genre_rock","key_0","key_1","key_2",
"key_3","key_4","key_5","key_6","key_7","key_8","key_9","key_10","key_11","mode_0","mode_1"]

def load_and_split(test_size=0.2, random_state=42):
    df = pd.read_csv("data/processed/spotify_songs_processed.csv")

    X = df[continuous_features+binary_features]
    y = df["updated_popularity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Standardize continuous features
    scaler = StandardScaler()
    X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
    X_test[continuous_features] = scaler.transform(X_test[continuous_features])

    return X_train, X_test, y_train, y_test