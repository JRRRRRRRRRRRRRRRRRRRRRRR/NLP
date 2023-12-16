from sentence_transformers import SentenceTransformer
import torch
import json

from helper import clean_lyrics, perform_sentiment_analysis, extract_topics


# STEP 1: Data cleaning
# Load the songs in a dictionary and clean unnecessary chararcters from lyrics
songs = clean_lyrics('handpicked_lyrics_short.jl')


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
for song in sentiment_topics:
    song_title = song['song_title']
    song_lyrics = songs[song_title]
    topics = extract_topics(song_lyrics)
    song['topics'] = topics


# STEPT 5: Save analysis
with open('sentiment_topics.json', 'w') as file:
    json.dump(sentiment_topics, file)

with open('songs.json', 'w') as file:
    json.dump(songs, file)