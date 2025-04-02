import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

# Load books dataset
books = pd.read_csv("books_with_emotions.csv")

# Handle missing values
books["description"] = books["description"].fillna("").astype(str)
categories = ["All"] + sorted(books["simple_categories"].dropna().astype(str).unique())

books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(), "cover-not-found.jpg", books["thumbnail"] + "&fife=w800"
)

# Load and process documents
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Use Hugging Face embeddings
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding=huggingface_embeddings)  # Fixed embedding argument


# def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None,
#                                       initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
#     recs = db_books.similarity_search_with_score(query, k=initial_top_k)
#
#     # Ensure unpacking tuple results (Document, Score)
#     recs = [rec[0] for rec in recs]  # Extract only the Document object
#
#     # Extract ISBNs safely
#     books_list = [rec.page_content.strip('"').split()[0] for rec in recs]
#     books_list = [isbn for isbn in books_list if isbn.isdigit()]  # Ensure valid ISBNs
#
#     book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)
#
#     if category != "All":
#         book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
#     else:
#         book_recs = book_recs.head(final_top_k)
#
#     # Sorting by emotional tone
#     tone_columns = {"Happy": "joy", "Surprising": "surprise", "Angry": "anger", "Suspenseful": "fear", "Sad": "sadness"}
#     if tone in tone_columns:
#         book_recs = book_recs.sort_values(by=tone_columns[tone], ascending=False)
#
#     return book_recs

def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None,
                                      initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)

    # Ensure unpacking tuple results (Document, Score)
    recs = [rec[0] for rec in recs]  # Extract only the Document object

    # Debugging print statements (inside function)
    print("Retrieved results:", recs)
    print("First item type:", type(recs[0])) if recs else print("No results found")

    # Extract ISBNs safely
    books_list = [rec.page_content.strip('"').split()[0] for rec in recs]
    books_list = [isbn for isbn in books_list if isbn.isdigit()]  # Ensure valid ISBNs

    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Sorting by emotional tone
    tone_columns = {"Happy": "joy", "Surprising": "surprise", "Angry": "anger", "Suspenseful": "fear", "Sad": "sadness"}
    if tone in tone_columns:
        book_recs = book_recs.sort_values(by=tone_columns[tone], ascending=False)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        truncated_description = " ".join(row["description"].split()[:30]) + "..."

        authors_split = row["authors"].split(";")
        authors_str = (
            f"{authors_split[0]} and {authors_split[1]}" if len(authors_split) == 2
            else f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}" if len(authors_split) > 2
            else row["authors"]
        )

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")
    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

if __name__ == "__main__":
    dashboard.launch(share=True, allow_flagging="never")