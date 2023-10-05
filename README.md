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
Main idea to follow LinkedIn Learning courses which are mostly free to us.
Resources
1. ? 