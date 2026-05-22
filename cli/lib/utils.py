from lib.inverted_index import InvertedIndex
from lib.sanitizer import sanitizer
from constants import BM25_K1, BM25_B

def load_stopwords() -> list:
    # Loads stopwords
    with open('data/stopwords.txt', 'r') as file:
            return file.read().splitlines()
    
def search_for_args(query):
    stopwords = load_stopwords()
    db = InvertedIndex()
    db.load()
    sanitized_search_arg_tokens = sanitizer(query, stopwords)
    results = []
    for token in sanitized_search_arg_tokens:
        try:
            results.extend(db.get_documents(token))
            if len(results) > 5:
                return results
        except Exception as e:
            print(f"Unable to find")
    return results

def bm25_idf_command(term: str) -> float:
    # Allows us to test the `get_bm25_idf` method
    db = InvertedIndex()
    db.load()
    bm25idf = db.get_bm25_idf(term)
    return bm25idf

def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B):
    db = InvertedIndex()
    db.load()
    stopwords = load_stopwords()
    sanitized_term = sanitizer(term, stopwords)
    bm25tf = db.get_bm25_tf(doc_id, sanitized_term[0], k1, b)
    return bm25tf