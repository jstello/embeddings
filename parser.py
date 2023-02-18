import nltk
import sys


TERMINALS = """
    Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
    Adv -> "down" | "here" | "never"
    Conj -> "and" | "until"
    Det -> "a" | "an" | "his" | "my" | "the"
    N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
    N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
    N -> "smile" | "thursday" | "walk" | "we" | "word"
    P -> "at" | "before" | "in" | "of" | "on" | "to"
    V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
    V -> "smiled" | "tell" | "were"
    """


NONTERMINALS = """ 
    S -> NP VP | NP VP Conj VP | NP ADVP
    NP -> N | Det NP | AP NP | N PP | Det NP | Adv Det N | Det NP Conj NP
    VP -> Adv V PP | V | V NP | V NP PP | V PP | V Det NP | VP ADVP | NP VP
    AP -> Adj | Adj AP
    ADVP -> NP PP | P NP | A VP | Adv VP | VP Adv | VP ADVP | ADVP ADVP | P NP P
    PP -> P NP | P VP
    PP -> P NP Adv
    VP -> VP Conj VP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    sentence = sentence.lower()
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize
    
    words = word_tokenize(sentence)
    
    [words.remove(word) for word in words if not any(char.isalpha() for char in word)]
    
    
    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    noun_phrase_chunks = []
    
    subtrees = tree.subtrees()
    
    # Subtrees that are labeled as NP
    subtrees_np = [subtree for subtree in subtrees if subtree.label() == "NP"] 
    
    
    # Subtrees that are labeled as NP and do not contain any other NP as a subtree
    noun_phrase_chunks = [st for st in subtrees_np if not any(st in st2 for st2 in subtrees_np if st2 != st)]
    
    return noun_phrase_chunks


if __name__ == "__main__":
    main()
