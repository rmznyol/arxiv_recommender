import arxiv
import pandas as pd
from tqdm import tqdm

# Define your search query
while True:
    query = input("search (e.g 'Minimal Surfaces'): ")
    if query:
        break

# Set the number of results you want to retrieve
while True:
    try:
        max_results = int(input("Maximum number of results requested: "))
    except ValueError:
        print("Please enter a positive integer number")
        continue
    else:
        break

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

# Search for articles on arXiv
search = arxiv.Search(
    query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
)

# Iterate through the search results and append data to the DataFrame
iterator_with_tqdm = tqdm(search.results())
for result in iterator_with_tqdm:
    iterator_with_tqdm.set_description(f"Loading {result.title} ...")
    article_info = {
        "Title": result.title,
        "Authors": ", ".join([author.name for author in result.authors]),
        "Published": result.published,
        "Abstract": result.summary,
        "Link": result.pdf_url,
        "PrimaryCategory": result.primary_category,
        "Categories": result.categories,
    }
    df = pd.concat([df, pd.DataFrame([article_info])], ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv("data/minimal_surfaces_articles.csv", index=False)
