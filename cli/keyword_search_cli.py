import argparse
import json
import math

from lib.inverted_index import InvertedIndex
from lib.sanitizer import sanitizer

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    build_parser = subparsers.add_parser("build", help="Build database")
    
    tf_parser = subparsers.add_parser("tf", help="Gets term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency of")
    
    idf_parser = subparsers.add_parser("idf", help="Calculate Inverse Document Frequency")
    idf_parser.add_argument("term", type=str, help="Term to get inverse document frequency of")

    args = parser.parse_args()
    
    # Load our dataset
    results = []
    with open('data/movies.json', 'r') as file:
        dataset = json.load(file)
    
    with open('data/stopwords.txt', 'r') as file:
        stopwords = file.read().splitlines()

    match args.command:
        case "build":
            # Build our Inverted Index
            db = InvertedIndex()
            db.build()
            
            # Save to list
            db.save()
            
            # TODO: Add logger statements on success?
            
        case "tf":
            # It should take a document ID and a term as arguments.
            # It should print the term frequency for that term in the document with the given ID.
            # If the term doesn't exist in that document, it should print "0".
            print(f"Counting {args.term} across document #{args.doc_id}")
            db = InvertedIndex()
            try:
                db.load()
            except Exception as e:
                print(f"Exception encountered: {e}")
                return
            try:
                term_count = db.get_tf(args.doc_id, args.term)
            except Exception as e:
                print(f"Exception encountered: {e}")
                return
            print(f"The term `{args.term}` appeared {term_count} time(s) in document #{args.doc_id}")

        
        case "search":
            # Perform the search
            print(f"Searching for: {args.query}")
            
            # Scan through movies
            movie_counter = 0
            
            # We're now going to use our InvertedIndex data set via load()
            db = InvertedIndex()
            try:
                db.load()
            except Exception as e:
                print(f"Exception encountered: {e}")
                return
            
            sanitized_search_arg_tokens = sanitizer(args.query, stopwords)
            results = []
            for token in sanitized_search_arg_tokens:
                try:
                    results.extend(db.get_documents(token))
                    if len(results) > 5:
                        break
                except Exception as e:
                    print(f"Unable to find")

            for result in results:
                movie_counter += 1
                print(f"{db.docmap[result]['id']}. {db.docmap[result]['title']}")
                if movie_counter == 5:
                    break 
                
        case "idf":
            # Calculate the IDF for a given term
            
            db = InvertedIndex()
            try:
                db.load()
            except Exception as e:
                print(f"Exception encountered: {e}")
                return
            
            total_doc_count = len(db.docmap)
            sanitized_term = sanitizer(args.term, stopwords)
            term_match_doc_count = len(db.index[sanitized_term[0]])
            
            # Calculate the IDF
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()