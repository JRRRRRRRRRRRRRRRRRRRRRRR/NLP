import json

from helper import find_most_similar_song, perform_sentiment_analysis, extract_topics, match_topics, find_all_similar_songs, lscore


# STEP 1: Load song lyrics and analysis from files
with open('sentiment_topics.json', 'r') as file:
    sentimen_topics = json.load(file)
with open('songs.json', 'r') as file:
    songs = json.load(file)

res = []
for song in songs:
    res.append(song)


# STEP 2: User questionaire
user_input_for_smanalysis = ''
user_input_for_topic_modelling = ''

# Ask questions and store the answers
question_0 = input("What's your test number?\n")
question_1 = input("How are you doing?\n")
user_input_for_smanalysis += question_1 + '.\n'

question_2 = input("If you were being completely honest with me, how would you describe your feelings lately?\n")
user_input_for_smanalysis += question_2 + '.\n'

question_3 = input("If you could choose a color to describe your current mood, what would it be?\n")
user_input_for_smanalysis += question_3 + '.\n'

question_4 = input("What word would you use to describe your life right now?\n")
user_input_for_topic_modelling += question_4 + '.\n'

question_5 = input("What did you do today?\n")
user_input_for_topic_modelling += question_5 + '.\n'

question_6 = input("What's a memory that has been influencing your emotions today?\n")
user_input_for_topic_modelling += question_6

# Variable that sums up all the users input (for embeddings)
user_input = user_input_for_smanalysis + user_input_for_topic_modelling


# STEP 3: Find best fitting song based on embeddings
best_embedding_match, similarity_score = find_most_similar_song(songs, user_input)
b_embed_matches = find_all_similar_songs(songs, user_input)


# STEP 4: Find best fitting song based on sentiment
user_sentiment_score = perform_sentiment_analysis(user_input)

s_res = []
for song in sentimen_topics:
    s_score = song['sentiment_score']
    sentiment_match = 1 - abs(s_score - user_sentiment_score) / 4
    s_res.append({'song': song['song_title'], 'sentiment_match': sentiment_match})

best_sentiment_matches = {entry['song']: entry['sentiment_match'] for entry in s_res}

best_sentiment_match = max(best_sentiment_matches, key=best_sentiment_matches.get)


# STEP 5: Find best fitting song based on topic analysis
topics = extract_topics(user_input_for_topic_modelling)
score_dict, best_topic_match = match_topics(topics, sentimen_topics)


# STEP 6: Find best fitting song based on combination of embeddings, sentiment and topics
best_combination_match = ''
maxi = 0
for song in songs:
    e = b_embed_matches[song]
    s = best_sentiment_matches[song]
    t = score_dict[str(res.index(song))] / 100
    if lscore(e,s,t) > maxi:
        maxi = lscore(e,s,t)
        best_combination_match = song
    

# STEP 7: Print the results
print(f"\n##### TEST NUMBER {question_0} #####")
print(question_1)
print(question_2)
print(question_3)
print(question_4)
print(question_5)
print(question_6)
print(f'the following topics were identified: {topics}')
print(f'based on embeddings, your song is: {best_embedding_match}')
print(f'based on sentiment, your song is: {best_sentiment_match}')
print(f'based on topics, your song is: {best_topic_match}')
print(f'based on everything, your song is: {best_combination_match}')