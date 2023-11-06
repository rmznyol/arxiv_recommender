import pandas as pd 

from string import punctuation 
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class Vectorizer:
    def __init__(self, use_lemmatizer = True):
        self.use_lemmatizer = use_lemmatizer
        self.trimmer = nltk.WordNetLemmatizer() if self.use_lemmatizer else nltk.PorterStemmer()
        # nltk.download('stopwords') # need to run this once
        self.stopword =  nltk.corpus.stopwords.words('english')
    
    def trim(self, text):
        return self.trimmer.lemmatize(text) if self.use_lemmatizer else self.trimmer.stem(text)
    
    def text_cleaner(self, text):
        text_nopunct = ''
        for char in text: # no reason to loop through twice just to use list comp
            if char not in punctuation:
                text_nopunct += char.lower()

        #tokenize it 
        tokens = re.split('\W+', text_nopunct) 

        # apply stemming or lemmatizing inside the first loop to avoid looping twice again
        tokens_no_stop_only_trimmed = [self.trim(word) for word in tokens if word not in self.stopword] 

        return tokens_no_stop_only_trimmed

    def vectorize(self,ds):
        tfidf_vect = TfidfVectorizer(analyzer=self.text_cleaner)
        sparse_matrix = tfidf_vect.fit_transform(ds)
        return sparse_matrix
        
    
def vectorize_and_save(unprocessed_data_file_path, column_to_vectorize):
    #get cleaner
    # if you need to use stemmer instead, put the input use_lemmatizer = False
    df = pd.read_csv(unprocessed_data_file_path)
    vectorizer = Vectorizer()
    # faster apply
    sparse_matrix = vectorizer.vectorize(df[column_to_vectorize])

    #save it in data folder 
    joblib.dump(sparse_matrix, f'data/{column_to_vectorize}_tfidf_sparse_matrix.pkl')
    return f'data/{column_to_vectorize}_tfidf_sparse_matrix.pkl'



if __name__ == '__main__':
    # get data
    file_name = 'data/minimal_surfaces_articles.csv'
    column_to_vectorize = 'Abstract'
    vectorize_and_save(file_name, column_to_vectorize)