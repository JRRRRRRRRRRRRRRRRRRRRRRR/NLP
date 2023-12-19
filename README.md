# Recommending Songs: Embeddings, Sentiment and Topics

This is a project for the TU Delft Course TI3160TU: Natural Language Processing.

Music plays a vital role in people's lives, exerting a significant influence on their emotions. Occasionally, certain songs resonate deeply with individuals, striking a chord at precisely the right moment, though these instances are rare. These moments can profoundly impact people's feelings, aiding them in processing their experiences. Hence, we wanted to recommend the perfect song to listeners at the most opportune time, enhancing their emotional journey and providing solace or inspiration as needed.

The project consists of a CLI questionnaire that captures the users' state, a [lyrics database from Genius](https://www.cs.cornell.edu/~arb/data/genius-expertise/), and an NLP pipeline with the following NLP techniques:
- Embeddings ([paraphrase-MiniLM-L6-V2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2))
- Sentiment Analysis ([bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment))
- Topic Analysis (GPT-3.5-turbo and GPT-4-turbo)

## Use

To use this project, you can clone the GitHub repository, make sure to add an api_keys.json file in the main repository and add your OpenAI API key there. The json file should have the following format: {"OPENAI_API_KEY": "YOUR_API_KEY_HERE"}.

## Files

The following files are in the project:
- **matching.py**: runs the CLI questionnaire and creates song recommendations (note: this can take 2-3 minutes due to GPT API requests).
- **process_lyrics.py**: performs embedding, sentiment, and topic analysis on the songs from the specified file (this file can be chosen in step 1 in process_lyrics.py). `lyrics_embeddings.pt`, `sentiment_topics.json`, and `songs.json` are the result of the processing. There are 3 subsets of the Genius dataset already in the repository:
	- **randomly_selected_lyrics.jl**: 50 randomly selected songs from the Genius dataset.
	- **handpicked_lyrics.jl**: 100 handpicked songs from the Genius, The songs are chosen to get diversity in genre, topics and sentiment.
	- **handpicked_lyrics_short.jl**: 50 randomly selected songs from handpicked_lyrics.jl (the topic analysis doesn't work properly for the bigger dataset).
- **helper.py**: contains all the helper functions: clean_lyrics, find_most_similar_song, find_all_similar_songs, perform_sentiment_analysis, extract_topics, match_topics, lscore
- **lyrics_data_analysis.ipynb**: A jupyter notebook that contains all the code to generate all the plots for the data analysis.
- **Data_analysis**: A folder with all the plots for the data analysis
- **NLP_likert.R**: RStudio code to nicely represent data from survey in a plot
- ****NLP_data****: Excel file containing data from the user test survey
