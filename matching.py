import json

from helper import find_most_similar_song, perform_sentiment_analysis


# STEP 1: Load song lyrics and analysis from files
with open('sentiment_topics.json', 'r') as file:
    sentimen_topics = json.load(file)
with open('songs.json', 'r') as file:
    songs = json.load(file)


# STEP 2: User questionaire
# TODO Fedde/Britt: implement the new questions
user_input = input('gimme\n')


# STEP 3: Find best fitting song based on embeddings
most_similar_song, similarity_score = find_most_similar_song(songs, user_input)


# STEP 4: Find best fitting song based on sentiment
user_sentiment_score = perform_sentiment_analysis(user_input)

current_song = None
difference = 1000
for song in sentimen_topics:
    sentiment_score = song['sentiment_score']
    if abs(sentiment_score - user_sentiment_score) < difference:
        difference = abs(sentiment_score - user_sentiment_score)
        current_song = song['song_title']

# STEP 5: Find best fitting song based on topic analysis
# TODO Koen: implement topic matching


# STEP 6: Find best fitting song based on combination of embeddings, sentiment and topics
# TODO Gijs: implement combining techniques


# STEP 7: Print the results
print(f'based on embeddings, your song is: {most_similar_song}')
print(f'based on sentiment, your song is: {current_song}')