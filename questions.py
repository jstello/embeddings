import nltk
import sys

FILE_MATCHES = 1
SENTENCE_MATCHES = 5


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    
    file_words = {filename: tokenize(files[filename]) for filename in files}
    
    file_idfs = compute_idfs(file_words)
    
    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):  # loop through each passage in the file  
            for sentence in nltk.sent_tokenize(passage): # its removing the 'l' from 'level' -> evel
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens
        print(f"Top file matches: {filenames}")

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print()
        print(match)

# Cache load_files and tokenize
import functools
@functools.lru_cache(maxsize=None)
def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    
    import glob
    import os
    import codecs
    import nltk
    # 

    # punkt
    nltk.download('punkt')
    # Get all files in directory using os.sep
    
    file_paths = glob.glob(os.path.join(directory, '*.txt'))
    files_dict = {}
    for file_path in file_paths:
        with open(file_path, 'r', encoding='UTF-8') as file:
            # the key should be just the file name, not the path
            key = os.path.basename(file_path)
            files_dict[key] = file.read()
    
    return files_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import nltk

    @functools.lru_cache(maxsize=None)
    def download_stopwords():
        nltk.download("stopwords", quiet=True)
    download_stopwords()
    
    tokens = word_tokenize(document)
    
    # Filter out stopwords
    tokens = [token for token in tokens if not token in stopwords.words('english')]
    
    # Filter out punctuation adn lowercase
    tokens = [token.lower() for token in tokens if token.isalpha()]

    
    return tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    import math
    
    idf_dict = {}
    
    # Inverse document frequency is the log of the number of documents 
    # divided by the number of documents that contain the word
    
    # import numpy as np
    
    num_docs = len(documents)
     
    # create set of all words in all documents
    
    words = set()
    
    for doc in documents:
        words.update(documents[doc])
        
    for word in words:
        num_docs_with_word = 0
        for doc in documents:
            if word in documents[doc]:
                num_docs_with_word += 1
        idf_dict[word] = math.log(num_docs / num_docs_with_word)
    
      
    return idf_dict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    
    # Create a dictionary to store the tf-idf scores of each file
    scores = {}
    
    # Calculate the tf-idf score for each file
    for filename, words in files.items():  # for each file in the corpus and its words
        tf_idf = 0
        for word in query:  # e.g. "python"
            if word in words: # if the word is in the article (python is not in the probability, or neural network articles)
                # Term frequency is the number of times the word appears in the article
                tf = words.count(word)
                idf = idfs[word]
                tf_idf += tf * idf
        scores[filename] = tf_idf  # finished checking query in first two file
    
    # Return a list of the filenames of the top n files, ranked by their tf-idf scores
    top_files_list = sorted(scores, key=scores.get, reverse=True)[:n]
    
    print(f"Top file matches: {top_files_list}"
          )
    return top_files_list


def top_sentences(query, sentences, idfs, n):
    # Create a dictionary to store the idf and query term density of each sentence
    scores = {}
    
    # Calculate the idf and query term density for each sentence
    for sentence, words in sentences.items():
        idf = 0
        qtd = 0
        for word in query:
            if word in words:
                idf += idfs[word]
                qtd += 1 / len(words)
        scores[sentence] = (idf, qtd)
    
    # Return a list of the top n sentences, ranked by idf and then by query term density
    return sorted(scores, key=lambda x: (scores[x][0], scores[x][1]), reverse=True)[:n]

if __name__ == "__main__":
    main()
