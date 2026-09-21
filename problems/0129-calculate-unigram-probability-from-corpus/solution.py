def unigram_probability(corpus: str, word: str) -> float:
    # Your code here
    total = len(corpus.split())
    counter = [x for x in corpus.split() if x == word]
    count = len(counter)
    return round(count/total, 4)