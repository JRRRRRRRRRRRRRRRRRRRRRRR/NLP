from sentence_transformers import SentenceTransformer
import torch
import json

from helper import clean_lyrics, perform_sentiment_analysis

# STEP 1: Data cleaning
# Load the songs in a dictionary and clean unnecessary chars from lyrics
songs = clean_lyrics('lyrics.jl')


# STEP 2: Embeddings
# Save a tensor with the embeddings for each song
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
song_embeddings = model.encode(list(songs.values()), convert_to_tensor=True)
torch.save(song_embeddings, 'lyrics_embeddings.pt')


sentiment_topics = []
    
# STEP 3: Sentiment
# Perform sentiment analysis on the songs
for song in songs:
    current_song = {}
    current_song['song_title'] = song
    score = perform_sentiment_analysis(songs[song])
    current_song['sentiment_score'] = score
    sentiment_topics.append(current_song)

# STEP 4: Topics
# Extract topics from the songs using GPT3.5
# TODO

with open('sentiment_topics.json', 'w') as file:
    json.dump(sentiment_topics, file)

with open('songs.json', 'w') as file:
    json.dump(songs, file)