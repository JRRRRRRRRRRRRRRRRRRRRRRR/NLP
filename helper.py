import re
import torch
from sentence_transformers import util, SentenceTransformer

from transformers import pipeline


def clean_lyrics(filename):
    """
    filename: file that contains the lyrics of all the songs

    ---
    Opens the songs from file and removes unnecessary characters/empty space.
    Returns the result in a dict[str, str] = {'song_name': 'song_lyrics', ...}
    """
    # Open the file using with
    with open(filename) as f:
        songs = {}

        # Loop through the songs and preprocess them
        for song in f:
            # Split the song into first and second part
            first, second = song.split(",", 1)
            _, song_name = first.split(":", 1)
            song_name = song_name.strip()
            song_name = song_name.strip('"')
            song_name = song_name[:len(song_name) - 7]

            # Split the second part into two parts, the second contains the lyrics
            _, song_lyrics = second.split(":", 1)
            song_lyrics = song_lyrics.replace("\\n", " ")
            song_lyrics = re.sub("\[.*?\]"," ", song_lyrics)

            # Make a list out of the lyrics
            lines = song_lyrics.split("    ")
            lines = [line.strip() for line in lines]
            lines = list(filter(None, lines))

            songs[song_name] = ' '.join(lines)
        return songs
    

def find_most_similar_song(songs, user_input):
    """
    model: pretrained embeddings model
    songs: dictionary with all the songs and lyrics
    user_input: string with users' reaction to a question
    
    ---
    Returns the song that's closest to the user input in the embedding space
    """
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    song_embeddings = torch.load('lyrics_embeddings.pt')
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, song_embeddings)[0]
    index = similarities.argmax().item()
    most_similar_song = list(songs.keys())[index]
    return most_similar_song, similarities[index].item()


def perform_sentiment_analysis(text):
    """
    text: string to perform the sentiment analysis on

    ---
    Returns the sentiment score on a scale from 1-5 
    """
    sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    
    # Split lyrics into chunks
    chunk_size = 1200
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # TODO: currently many 1 and 5 starts, check whether this comes from chunking
    # Analyze sentiment for each chunks
    results = []
    for chunk in chunks:
        result = sentiment_analyzer(chunk)
        results.append(result[0])
    
    # Combine sentiment results
    average_score = sum(float(result['label'].split()[0]) for result in results) / len(results)
    
    return average_score