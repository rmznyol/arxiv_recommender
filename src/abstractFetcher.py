import arxiv
import pandas as pd
from tqdm import tqdm


def fetch_and_save(query:str, max_results:int):
    #check if it is a valid query
    query_check(query, max_results)

    # Search for articles on arXiv
    search = arxiv.Search(
    query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )

    #save your data as a dataframe
    file_path_to_save = f'data/{query.strip().replace(" ", "_").lower()}_articles.csv'
    save_data(search, file_path_to_save, max_results)

    return file_path_to_save
# Define your search query
def query_check(query:str ,max_results: int):
    if not query:
        raise ValueError('empty query')
    if type(query) != str:
        raise TypeError('please enter a string of query')
    if not max_results != int:
        raise TypeError('Please enter a positive integer number')
    if max_results < 0:
        raise ValueError('please enter a positive integer')
    return query, max_results

def save_data(search, file_path_to_save, max_results):
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
    df.to_csv(file_path_to_save, index=False)

if __name__ == '__main__':
    # get data
    fetch_and_save('Minimal Surfaces', 20)