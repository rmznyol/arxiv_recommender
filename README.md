# Welcome to arXiv recommender system project

### Main Suggestions and TODOs:
- Follow the 'dev' branch until finalized
- Use black to reformat files before updating: 
    when you create & update a file `<file_name>`
    - first run `black <file_name>`
    - then add and commit the file.
- Stick to Modular programming as much as possible (every 'function' should have a script and class if necessary)
- also add Learning Branch

### How to fetch article data and more:
1. Clone this repo
2. Create a conda env and install requirements: `conda create --name <env> --file requirements.txt`
3. run `$python abstractFetcher.py`, which will ask for your query and how many articles you wish to save. (searched by relevance)
4. Find the resulting data under the `data` directory with the name `<query>_articles.csv`. 

### Learning List:
In order to create a nlp - based content recommender system, according to ChatGPT, we roughly need to learn the following: 
1. Natural Language Processing (NLP):
    - Text Preprocessing: Learn how to clean and preprocess textual data, including tasks like tokenization, stemming, lemmatization, and stop-word removal.
    - Text Representation: Understand different methods for representing text data, such as bag-of-words (BoW), TF-IDF (Term Frequency-Inverse Document Frequency), and word embeddings (Word2Vec, GloVe).
    - NLP Libraries: Familiarize yourself with NLP libraries and frameworks like NLTK, spaCy, and the Hugging Face Transformers library.
2. Recommendation Algorithms:
    - Content-Based Filtering: Understand how to recommend items based on their attributes and the user's historical preferences.
    - Collaborative Filtering: Learn about collaborative filtering techniques, including user-based and item-based methods.
    - Matrix Factorization: Explore matrix factorization methods like Singular Value Decomposition (SVD) and Alternating Least Squares (ALS).
    - Hybrid Models: Study how to combine different recommendation algorithms to improve accuracy, such as hybrid content-collaborative systems.

Current suggested learning path:
1. https://www.linkedin.com/learning/nlp-with-python-for-machine-learning-essential-training
2. https://www.linkedin.com/learning/building-recommender-systems-with-machine-learning-and-ai (Probably slightly outdated in terms of which libraries they use):
