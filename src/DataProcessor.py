import numpy as np
import pandas as pd
from src.abstractFetcher import fetch_and_save
from src.vectorizer import vectorize_and_save

# TASKS:

# 1: convert primary category to numbers
# 2: convert categories to numbers
# 3: vectorize and save abstracts
# 4: vectorize and save titles
# 5: write a function to do the same thing to user


def ProcessData(query, max_results):
    # if query is already run, skip this step
    unprocessed_data_file_path = fetch_and_save(query, max_results)

    print(f"Raw data installed @ {unprocessed_data_file_path}")

    # column_to_vectorize = "Abstract"
    # vectorized_data_file_path, fitted_vectorizer = vectorize_and_save(
    #     unprocessed_data_file_path, column_to_vectorize
    # )

    # 1: TODO
    primary_category_data = pd.read_csv(unprocessed_data_file_path, usecols=[5])
    primary_category_array = np.zeros(primary_category_data.values.shape)
    np.save("data/primary_category_array.npy", primary_category_array)

    # 2: TODO
    categories_data = pd.read_csv(unprocessed_data_file_path, usecols=[6])
    categories_array = np.zeros(categories_data.values.shape)
    np.save("data/categories_array.npy", categories_array)

    # 3:
    vectorized_abstract_data_file_path, fitted_abstract_vectorizer = vectorize_and_save(
        unprocessed_data_file_path, "Abstract"
    )
    print(
        f"Vectorized Abstract data is successfully saved at @ {vectorized_abstract_data_file_path}"
    )
    # 4:
    vectorized_title_data_file_path, fitted_title_vectorizer = vectorize_and_save(
        unprocessed_data_file_path, "Title"
    )
    print(
        f"Vectorized Title data is successfully saved at @ {vectorized_title_data_file_path}"
    )


if __name__ == "__main__":
    ProcessData("Ramazan Yol", 50)
