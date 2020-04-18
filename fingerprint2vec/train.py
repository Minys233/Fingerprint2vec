from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import argparse
from randomwalk_corpus import RandomWalk
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    inputf = args.input
    outputf = args.output
    sentenses = LineSentence(inputf)

    model = Word2Vec(sentences=sentenses, size=300, window=10, min_count=3, sg=1, negative=15, workers=16)
    model.save(outputf)


