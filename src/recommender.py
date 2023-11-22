import numpy as np
import pandas as pd
import joblib
from heapq import heappush, heappushpop
from src.abstractFetcher import fetch_user_data

NUMBER_OF_SUGGESTIONS = 5
MAX_USER_ARTICLE_COUNT = 5


def cosine_similarity(vector_a, vector_b):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    - vector_a (scipy.sparse._csr.csr_matrix): The first vector.
    - vector_b (scipy.sparse._csr.csr_matrix): The second vector.

    Returns:
    float: Cosine similarity between the two vectors. The value is in the range [-1, 1],
    where 1 indicates identical vectors, 0 indicates orthogonal vectors, and -1 indicates
    diametrically opposed vectors.

    Notes:
    - if one of the vectors is zero then raises an error
    - Cosine similarity is calculated as the dot product of the vectors divided by
      the product of their Euclidean norms.

    """
    dot_product = sparse_dot(vector_a, vector_b)
    norm_a = sparse_norm(vector_a)
    norm_b = sparse_norm(vector_b)

    if not norm_a or not norm_b:
        return 0
        # raise ValueError("one of the vectors is zero")

    similarity = dot_product / (norm_a * norm_b)

    return similarity


def sparse_dot(vector_a, vector_b):
    result = vector_a * vector_b.T
    # if two vectors are orthogonal then the output vector is empty
    # thus we need to handle it here seperately
    return result.data[0] if result else 0


def sparse_norm(vector_a):
    return np.sqrt(sparse_dot(vector_a, vector_a))


class ArxivRecommender:
    def __init__(
        self,
        feature_weights_array=[0, 0, 0.8, 0.2],
        authors=pd.read_csv("data/recorded_articles.csv", header=0, usecols=[1]),
        primary_category_array=np.load("data/primary_category_array.npy"),
        categories_array=np.load("data/categories_array.npy"),
        abstract_vector_data=joblib.load("data/Abstract_tfidf_sparse_matrix.pkl"),
        fitted_abstract_vectorizer=joblib.load("data/Abstract_tfidf_vectorizer.pkl"),
        title_vector_data=joblib.load("data/Title_tfidf_sparse_matrix.pkl"),
        fitted_title_vectorizer=joblib.load("data/Title_tfidf_vectorizer.pkl"),
        similarity_metric=cosine_similarity,
    ) -> None:
        """
        feature_weights_array shape needs care
        feature_weights_array = [.1, .05, .8, .01] primary, Categories, abstract, title
        """
        self.authors = authors
        self.similarity_metric = similarity_metric
        self.feature_weights_array = feature_weights_array

        self.primary_category_array = primary_category_array
        self.categories_array = categories_array
        self.number_of_stored_data_points = len(self.categories_array)

        self.abstract_vector_data = abstract_vector_data
        self.abstract_vectorizer = fitted_abstract_vectorizer
        self.title_vector_data = title_vector_data
        self.title_vectorizer = fitted_title_vectorizer

        self.number_of_suggestions: int = NUMBER_OF_SUGGESTIONS
        self.max_user_article_count: int = MAX_USER_ARTICLE_COUNT

    def recommend_to(self, user_name):
        # calculate similarity
        # return an top k
        # will inherit multiple features from both user_array and data_array
        # category_features + abstract_cosine_similarity + title_cosine_similarity
        # will use different weights
        # use a heap to keep a tab
        # also different articles will come
        """
        user_array shape? if it is single we need to reshape it
        feature_weights_array shape needs care
        feature_weights_array = [.1, .05, .8, .01] primary, Categories, abstract, title
        """
        # user will have k papers and when compared to an article in the dataset
        # we will choose score it with by the highest similarity among k
        recommendations = []  # Recommendation and its score needs to be stored

        (
            user_primary_category_data,
            user_categories_data,
            user_abstract_matrix,
            user_title_matrix,
        ) = self.process_user_data(user_name)

        for index in range(self.number_of_stored_data_points):
            if not self._is_user(index, user_name):
                score = self._get_article_score(
                    user_primary_category_data,
                    user_categories_data,
                    user_abstract_matrix,
                    user_title_matrix,
                    index,
                )
                if index < self.number_of_suggestions:
                    heappush(recommendations, (score, index))
                else:
                    heappushpop(recommendations, (score, index))

        # we need to sort it by similarity and rank them bc heap isnt sorted
        recommendations.sort(key=lambda x: x[0], reverse=True)

        recommendations_display = self._indices_to_articles(recommendations)
        return recommendations_display

    def _is_user(self, index, user_name):
        article_authors = self.authors.iloc[index].Authors

        return [
            author
            for author in article_authors.split(",")
            if author.strip() == user_name
        ]

    def _get_article_score(
        self,
        user_primary_category_data,
        user_categories_data,
        user_abstract_matrix,
        user_title_matrix,
        article_index,
    ):
        # Calculate similarity
        # TODO category info inclusion
        user_PrimaryCategory_score = self._get_Category_score()
        user_Categories_score = self._get_Category_score()
        abstract_similarity_score = self._get_similarity_score(
            user_abstract_matrix, self.abstract_vector_data[article_index]
        )
        title_similarity_score = self._get_similarity_score(
            user_title_matrix, self.title_vector_data[article_index]
        )

        # get the total raw score
        raw_scores = np.array(
            [
                user_PrimaryCategory_score,
                user_Categories_score,
                abstract_similarity_score,
                title_similarity_score,
            ]
        )

        # calculate the weigthed scire
        score = np.dot(raw_scores, self.feature_weights_array)
        return score

    def _get_similarity_score(self, user_array, data_vector):
        max_similarity = 0
        for user_vector in user_array:
            max_similarity = max(
                max_similarity, self.similarity_metric(user_vector, data_vector)
            )
        return max_similarity

    @staticmethod
    def _get_Category_score():
        # multiple as a vector  or
        # it is 1 if same else 0
        # needs normalization
        # TODO
        return 0

    def process_user_data(
        self,
        user_name,
        max_user_article_count=MAX_USER_ARTICLE_COUNT,
    ):

        user_data = fetch_user_data(user_name, max_user_article_count)
        # TODO change category data
        # user_PrimaryCategory = self._get_user_Category(user_data)
        # user_Categories = self._get_user_Category(user_data)

        user_primary_category_data = (np.zeros(user_data.PrimaryCategory.values.shape),)
        user_categories_data = (np.zeros(user_data.Categories.values.shape),)

        user_abstract_matrix = self.abstract_vectorizer.transform(user_data.Abstract)
        user_title_matrix = self.title_vectorizer.transform(user_data.Title)

        user_output = (
            user_primary_category_data,
            user_categories_data,
            user_abstract_matrix,
            user_title_matrix,
        )
        return user_output

    @staticmethod
    def _get_user_Category(user_data):
        # get the most frequent
        # convert it to array
        ...

    def _indices_to_articles(self, recommendations):
        print(len(recommendations))
        rows_to_keep = [0] + [index + 1 for _, index in recommendations]
        exclude = [
            i
            for i in range(self.number_of_stored_data_points + 1)
            if i not in rows_to_keep
        ]
        output = pd.read_csv("./data/recorded_articles.csv", skiprows=exclude, header=0)
        output["Similarity Scores"] = [
            similarity_score for similarity_score, _ in recommendations
        ]
        return output


if __name__ == "__main__":
    engine = ArxivRecommender()
    print(engine.recommend_to("Ramazan Yol"))
