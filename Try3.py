import re
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Load the pre-trained BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Open the file using with
with open("lyrics.jl") as f:
    # Make a dictionary for the songs
    songs = {}

    # Loop through the songs and preprocess them
    for song in f:
        # Split the song into first and second part
        first, second = song.split(",", 1)
        meh, song_name = first.split(":", 1)
        song_name = song_name.strip()
        song_name = song_name.strip('"')
        song_name = song_name[:len(song_name) - 7]

        # Split the second part into two parts, the second contains the lyrics
        meh, song_lyrics = second.split(":", 1)
        song_lyrics = song_lyrics.replace("\\n", " ")
        song_lyrics = re.sub("\[.*?\]"," ", song_lyrics)

        # Make a list out of the lyrics
        lines = song_lyrics.split("    ")
        lines = [line.strip() for line in lines]
        lines = list(filter(None, lines))
        songs[song_name] = ' '.join(lines)

# Create BERT embeddings for each song
song_embeddings = model.encode(list(songs.values()), convert_to_tensor=True)

# Function to find the most similar song based on user input (Slice top #)
def find_most_similar_song(user_input):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, song_embeddings)[0]
    index = similarities.argmax().item()
    most_similar_song = list(songs.keys())[index]
    return most_similar_song, similarities[index].item()

def perform_sentiment_analysis(song_lyrics):
    sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    
    # Split lyrics into chunks
    chunk_size = 300
    chunks = [song_lyrics[i:i+chunk_size] for i in range(0, len(song_lyrics), chunk_size)]
    
    # Analyze sentiment for chunks
    results = []
    for chunk in chunks:
        result = sentiment_analyzer(chunk)
        results.append(result[0])

    # Combine sentiment results
    average_score = sum(result['score'] for result in results) / len(results)
    overall_sentiment = "POSITIVE" if average_score >= 0.5 else "NEGATIVE"
    
    return overall_sentiment, average_score
    
# Sentiment
sentiment_results = []
for song in songs:
    current_song = {}
    current_song['song_title'] = song
    sentiment, score = perform_sentiment_analysis(song)
    current_song['sentiment'] = sentiment
    current_song['sentiment_score'] = score
    sentiment_results.append(current_song)

# User input
user_input = input('gimme\n')

most_similar_song, similarity_score = find_most_similar_song(user_input)

user_sentiment, user_sentiment_score = perform_sentiment_analysis(user_input)

current_song = None
difference = 1000
for song in sentiment_results:
    sentiment_score = song['sentiment_score']
    sentiment = song['sentiment']
    if user_sentiment == sentiment:
        if abs(sentiment_score - user_sentiment_score) < difference:
            difference = abs(sentiment_score - user_sentiment_score)
            current_song = song['song_title']

print(f'based on embeddings, your song is: {most_similar_song}')
print(f'based on sentiment, your song is: {current_song}')

# Github J
# Separate pre-processing and user input J
# Sentiment fixen K
# Topic modelling (OpenAI? pls) K
    # Integrate topic modelling
# Figure out how to combine all methods G
# Think of optimal questions to extract emotion B
    # What word would you use to describe your life right now
# Make multiple questions thing (3-5) F

# Next time
# Set up user-tests
# Do said user-tests

# Small analysis of dataset