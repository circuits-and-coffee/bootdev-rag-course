from lib.inverted_index import InvertedIndex



def bm25_idf_command(term: str) -> float:
    # Allows us to test the `get_bm25_idf` method
    
    db = InvertedIndex()
    db.load()
    bm25idf = db.get_bm25_idf(term)
    return bm25idf
    