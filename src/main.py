from abstractFetcher import fetch_and_save
from vectorizer import vectorize_and_save
import pandas as pd 
import joblib

query = 'Minimal Surfaces'
max_results = 20 

# # if query is already run, skip this step 
# unprocessed_data_file_path = fetch_and_save(query, max_results)

# print(unprocessed_data_file_path)

# column_to_vectorize = 'Abstract'
# vectorized_data_file_path = vectorize_and_save(unprocessed_data_file_path,
#                                                column_to_vectorize)


vectorized_data_file_path = 'data/Abstract_tfidf_sparse_matrix.pkl'
tfidf_sparse_matrix = joblib.load(vectorized_data_file_path)

print(tfidf_sparse_matrix.shape)

vectors_df = pd.DataFrame(tfidf_sparse_matrix.toarray())
print(vectors_df.head(10))