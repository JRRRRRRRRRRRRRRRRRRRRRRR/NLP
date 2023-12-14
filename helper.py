import re
import torch
from sentence_transformers import util, SentenceTransformer
from openai import OpenAI
import json
from transformers import pipeline
import os

try:
    with open('api_keys.json', 'r') as file:
        keys = json.load(file)
    os.environ["OPENAI_API_KEY"]  = keys['OPENAI_API_KEY']
except FileNotFoundError:
    print('There is no api_keys.json file. please add one.')
    os.environ["OPENAI_API_KEY"]  = keys['OPENAI_API_KEY']
except KeyError:
    print('There is no OpenAI API key provided in  api_keys.json')
os.environ["TOKENIZERS_PARALLELISM"] = 'true'


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


def find_all_similar_songs(songs, user_input):
    """
    model: pretrained embeddings model
    songs: dictionary with all the songs and lyrics
    user_input: string with users' reaction to a question
    
    ---
    Returns a dictionary containing all songs and their similarities to the user input
    """
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    song_embeddings = torch.load('lyrics_embeddings.pt')
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, song_embeddings)[0]
    
    # Create a dictionary with song names as keys and similarity values as values
    similar_songs = {song: similarity.item() for song, similarity in zip(songs.keys(), similarities)}
    
    return similar_songs


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

def extract_topics(text):
    """
    TODO
    """
    topic_extraction_prompt = f"""
    Analyze the following text and extract 5 main topics.
    Each topic should be no more than two words.
    Provide the topics in JSON format under the key 'topics'.

    Text: "{text}"

    Expected Output Format: 
    {{
      "topics": ["topic 1", "topic 2", "topic 3", "topic 4", "topic 5"]
    }}
    """
    client = OpenAI()
    try:
        topics = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": topic_extraction_prompt}
            ],
            temperature=0
        )
        response = topics.choices[0].message.content.strip()

        # Extracting JSON part from the response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_response = response[json_start:json_end]
            topics_dict = json.loads(json_response)
            topiclist = topics_dict.get("topics")
            if isinstance(topiclist, list):
                return topiclist
            else:
                raise TypeError("Extracted topics are not in list format.")
        else:
            raise ValueError("No JSON response found in the output.")
    except json.JSONDecodeError:
        print("Error parsing JSON from the response.")
        print("Response: ", response)
    except (KeyError, ValueError) as e:
        print(e)
        print("Response: ", response)
    except TypeError as e:
        print(e)
        print("Response: ", response)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Response: ", response)
    print('Provided non-sense topics')
    return ['', '', '', '', '']


def match_topics(topics_list, songs):
    """
    topics_list: List[str]. A list of topics provided by the user.
    songs: List[dict]. A list of dictionaries, each representing a song and containing a list of topics.

    ---
    Returns tuple containing two elements:
    - score_dict: Dict[int, float]. A dictionary where each key is an index from the songs list, and the value is the corresponding matching score.
    - best_match: str. The title of the song that best matches the user's topics, identified by the highest score in score_dict.
    """
    dataset = ""
    for i, song in enumerate(songs):
        dataset += f"{i}. {str(song['topics'])} \n"
    
    topic_matching_prompt = f"""
    Given a user's list of topics '{str(topics_list)}' and a dataset of items with their respective topics, I will evaluate and score each item based on thematic similarity and direct topic matches.

    For each item, I'll analyze its topics in relation to the user's topics. I'll first provide a brief explanation considering both thematic similarity and direct matches, and then assign a score from 0 to 100 based on this analysis, with a higher score for more closely related or directly matching topics.

    Dataset of Items and Their Topics:
    {dataset}

    Evaluation Process:
    - Note the topics for each item.
    - Explain the reasoning for the score, highlighting any direct matches or thematic similarities.
    - Assign a score based on the explanation.
    - Determine the single item with the best match

    Then, I'll compile the evaluations into a JSON format, indicating the scores and the 'best_match'.

    Let's start the evaluation:

    [The model will go through each item, explain the reasoning based on similarity and direct matches, and then assign a score. For example:]
    - Item 0's topics: ['love', 'night']. Reasoning: Good thematic similarity with 'dance' and 'club', but no direct match. Score: 70.
    - Item 1's topics: ['dance', 'joy']. Reasoning: Direct match with 'dance' and thematically similar to 'club'. Score: 90.

    [This process continues for all items]

    After evaluating all items:

    JSON Output:
    {{
      0: 70,
      1: 90,
      ...
      'best_match': index of the single item with the highest score as int
    }}
    """

    client = OpenAI()
    try:
        topics = client.chat.completions.create(
            model= "gpt-4-1106-preview",
            messages=[
                {"role": "user", "content": topic_matching_prompt}
            ],
            temperature=0
        )
        response = topics.choices[0].message.content.strip()

        # Extracting JSON part from the response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_response = response[json_start:json_end]
            score_dict = json.loads(json_response)
            best_match_index = score_dict['best_match']
            best_match = songs[best_match_index]['song_title']
            return score_dict, best_match
        else:
            raise ValueError("No JSON response found in the output.")
    except json.JSONDecodeError:
        print("Error parsing JSON from the response.")
        print("Response: ", response)
    except (KeyError, ValueError) as e:
        print(e)
        print("Response: ", response)
    except TypeError as e:
        print(e)
        print("Response: ", response)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Response: ", response)

def lscore(e, s, t):
    """
    TODO
    """
    return e + 3 * s + t