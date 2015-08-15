import numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

import config

__author__ = 'chetannaik'

FILTER_WORDS = []


def get_lemma(entry):
    wnl = WordNetLemmatizer()
    return str(wnl.lemmatize(entry.strip().lower()))


def get_filter_words():
    process_synsets = wn.synsets('process')
    global FILTER_WORDS
    for p_s in process_synsets:
        FILTER_WORDS.extend(p_s.lemma_names())
    FILTER_WORDS = map(lambda x: str(x), set(FILTER_WORDS))


def has_filter_keyword(word_list):
    for word in word_list:
        if get_lemma(word) in FILTER_WORDS:
            return True
    return False


def remove_filter_words(input_string):
    return_list = []
    input_list = map(lambda x: x.strip(), input_string.split("|"))
    for word in input_list:
        lemmatized_word = get_lemma(word)
        words = [lemmatized_word]
        words.extend(lemmatized_word.split())
        words = set(words)
        if not words & set(FILTER_WORDS):
            return_list.append(word)
    return " | ".join(return_list)


def filter_score_for_logging(score):
    if not np.isnan(float(score)):
        return str(score)
    else:
        return str(np.nan)


def generate_experiment_scores(experiment):
    roles = config.ROLES[experiment]
    config.SCORES.extend(map(lambda x: x + '_SCORE', roles))


def main():
    get_filter_words()
    input_string = "physical process | changes"
    print "Input String : {}".format(input_string)
    print "Return String: {}".format(remove_filter_words(input_string))


if __name__ == '__main__':
    main()
