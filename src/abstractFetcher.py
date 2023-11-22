import arxiv
import pandas as pd
from tqdm import tqdm


def fetch_and_save(query: str, max_results: int, save_it_by_name = False):

    df = fetch_data(query, max_results)

    if save_it_by_name:
        file_path_to_save = f'./data/{query.strip().replace(" ", "_").lower()}_articles.csv'
    else:
        file_path_to_save = './data/recorded_articles.csv'

    df.to_csv(file_path_to_save, index=False)

    return file_path_to_save


def fetch_data(query: str, max_results: int, sort_by_date=False):
    # check if it is a valid query
    _query_check(query, max_results)

    # Search for articles on arXiv
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
        # The following does not work well, it priortizes the date rathen than the author
        # sort_by=arxiv.SortCriterion.LastUpdatedDate if sort_by_date else arxiv.SortCriterion.Relevance
    )
    return _data_to_df(search, max_results)


def fetch_user_data(
    user_name,
    max_user_article_count,
):

    user_data = fetch_data(user_name, max_user_article_count, sort_by_date=True)
    filter = user_data.Authors.apply(
        lambda x: user_name in [author.strip() for author in x.split(",")]
    )
    user_data = user_data[filter]
    return user_data


# Define your search query
def _query_check(query: str, max_results: int):
    if not query:
        raise ValueError("empty query")
    if type(query) != str:
        raise TypeError("please enter a string of query")
    if not max_results != int:
        raise TypeError("Please enter a positive integer number")
    if max_results < 0:
        raise ValueError("please enter a positive integer")
    return query, max_results


def _data_to_df(search, max_results):
    # Create an empty DataFrame to store the data
    df = pd.DataFrame(
        columns=[
            "Title",
            "Authors",
            "Published",
            "Abstract",
            "Link",
            "PrimaryCategory",
            "Categories",
        ]
    )

    # Iterate through the search results and append data to the DataFrame
    iterator_with_tqdm = tqdm(search.results(), total=max_results)
    for result in iterator_with_tqdm:
        iterator_with_tqdm.set_description(f"Saving {result.title[:20]} ...")
        article_info = {
            "Title": result.title,
            "Authors": ", ".join([author.name for author in result.authors]),
            "Published": result.published,
            "Abstract": result.summary,
            "Link": result.pdf_url,
            "PrimaryCategory": result.primary_category,
            "Categories": result.categories,
        }
        df_article = pd.DataFrame([article_info])
        df = pd.concat(
            [df_iter for df_iter in [df, df_article] if not df_iter.empty],
            ignore_index=True,
        )
    # Save the DataFrame to a CSV file
    return df


if __name__ == "__main__":
    # get data
    fetch_and_save("Minimal Surfaces", 20)
