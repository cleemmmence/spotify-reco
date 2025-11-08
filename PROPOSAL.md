**Title** : Modeling Song Popularity with Supervised Learning and Audio Feature Analysis

**Category** : Data Analysis & Visualization

**Problem Statement & Motivation** : Music streaming platforms like Spotify rely on algorithms to understand listener preferences and songs that users are likely to enjoy, while also considering popularity trends. Understanding what drives a song’s popularity can provide valuable insights for artists, producers, and streaming platforms. This project aims to forecast song popularity using supervised machine learning models, including Linear Regression, Random Forest, and XGBoost, based on audio features. By analyzing which features most influence popularity, we can gain a deeper understanding of musical trends and listener preferences. This project shows how data science can guide strategic choices in the music industry by showcasing real-world applications of predictive modelling in a dataset of over 30'000 songs.

**Planned Approach & Technologies** : I will use a Kaggle dataset containing more than 30'000 songs, which includes audio and musical characteristics such as danceability, energy, loudness, speechiness, acousticness, valence, and tempo. To enrich this dataset, I will use the Spotipy Python library (Spotify Web API) to retrieve any missing or complementary audio features. This will ensure that all relevant variables are available for analysis and modelling.
After cleaning & analyzing the dataset, I will build supervised machine learning models to predict a song’s popularity. The models I will apply include Linear Regression, Random Forest Regression, and XGBoost Regression. I will compare their performance to determine which method best captures the relationship between audio features and popularity.
Model reliability will be evaluated using a train-test split (80/20) and evaluation metrics such as R², Root Mean Square Error, and Mean Absolute Error.

**Expected challenges and how you’ll address them** : One challenge will be the limit on the number of tracks I can query via the Spotify API, which could prevent me from enriching each entry. To overcome this, I will focus on the most relevant or missing features and work with a smaller, well-defined subset if necessary.
Another challenge is ensuring that the model is not overfitted. I will address this by using training-test splitting and cross-validation to check the model's performance on unseen data.


**Success criteria** : The project will be considered successful if the machine learning models are able to predict song popularity with reasonable accuracy on the test dataset. Specifically, I will look for a positive R² score and relatively low RMSE and MAE, showing that the model captures meaningful relationships between audio features and popularity.

**Stretch goals** : If time permits, I will extend the project by adding a simple song recommendation component based on feature similarity, to show how popularity prediction can complement music discovery. I may also try additional models or tuning methods to improve prediction performance, or create visual dashboards to make results easier to interpret.