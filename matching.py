import json

from helper import find_most_similar_song, perform_sentiment_analysis, extract_topics, match_topics


# STEP 1: Load song lyrics and analysis from files
with open('sentiment_topics.json', 'r') as file:
    sentimen_topics = json.load(file)
with open('songs.json', 'r') as file:
    songs = json.load(file)


# STEP 2: User questionaire
# Two strings. One stores input for semantic analysis and the second for topic modelling.
user_input_for_smanalysis = ''
user_input_for_topic_modelling = ''

# Ask three questions and store the answers
question_1 = input("How are you doing?\n")
user_input_for_smanalysis += question_1 + ' '

question_2 = input("If you were being completely honest with me, how would you describe your feelings lately?\n")
user_input_for_smanalysis += question_2 + ' '

question_3 = input("If you could choose a color to describe your current mood, what would it be?\n")
user_input_for_smanalysis += question_3 + ' '

question_4 = input("What word would you use to describe your life right now?\n")
user_input_for_topic_modelling += question_4 + ' '

question_5 = input("What did you do today?\n")
user_input_for_topic_modelling += question_5 + ' '

question_6 = input("What's a memory that has been influencing your emotions today?\n")
user_input_for_topic_modelling += question_6 + ' '



#Variable that sums up all the users input
user_input = user_input_for_smanalysis + user_input_for_topic_modelling


# STEP 3: Find best fitting song based on embeddings
best_embedding_match, similarity_score = find_most_similar_song(songs, user_input)


# STEP 4: Find best fitting song based on sentiment
user_sentiment_score = perform_sentiment_analysis(user_input)

difference = 1000
for song in sentimen_topics:
    sentiment_score = song['sentiment_score']
    if abs(sentiment_score - user_sentiment_score) < difference:
        difference = abs(sentiment_score - user_sentiment_score)
        best_sentiment_match = song['song_title']

# STEP 5: Find best fitting song based on topic analysis
topics = extract_topics(user_input_for_topic_modelling)
print(topics)
"""
FOR TESTING:

If you don't have an OpenAI API key, you can test with the following values.
topics = ['smiling', 'school', 'horror movie', 'interview', 'athletics']
score_dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0, '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 0, '24': 0, '25': 0, '26': 0, '27': 0, '28': 0, '29': 0, '30': 0, '31': 0, '32': 0, '33': 0, '34': 0, '35': 0, '36': 0, '37': 0, '38': 0, '39': 0, '40': 0, '41': 0, '42': 0, '43': 0, '44': 0, '45': 0, '46': 0, '47': 0, '48': 0, '49': 0, 'best_match': 0}
best_topic_match = 'Childish-gambino-heartbeat'
"""
score_dict, best_topic_match = match_topics(topics, sentimen_topics)
print(score_dict)
print(best_topic_match)


# STEP 6: Find best fitting song based on combination of embeddings, sentiment and topics
# TODO Gijs: implement combining techniques


# STEP 7: Print the results
print(f'based on embeddings, your song is: {best_embedding_match}')
print(f'based on sentiment, your song is: {best_sentiment_match}')
print(f'based on topics, your song is: {best_topic_match}')