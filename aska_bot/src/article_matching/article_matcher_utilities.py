import os
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import StanfordNERTagger

from .wikipedia_api import WikipediaAPI

# set environment var with path to stanford-ner.jar
os.environ["CLASSPATH"] = "../../models/stanford-ner.jar"


class PageMatcher:
    # instantiate stanford ner
    stanford_ner = StanfordNERTagger("../../models/english.conll.4class.distsim.crf.ser.gz")

    @staticmethod
    def process_query(
            query,
            remove_possessive=True,
            remove_punctuation=True,
            remove_stopwords=True
    ):
        """
        Performs basic string processing such as removing possessive, punctuation, and stopword
            as well as word tokenization.
        """
        if remove_possessive:
            # remove possessive
            query = query.replace("'s", "")
            query = query.replace("s'", "")

        if remove_punctuation:
            # remove punctuation
            query = query.translate(str.maketrans("", "", string.punctuation))

        if remove_stopwords:
            # remove stopwords
            eng_stopwords = stopwords.words('english')
            for word in eng_stopwords:
                if re.search(r"\\s+{}\\s+|{}\\s+".format(word, word), query):
                    query = query.replace(r"{} ".format(word), "")

        # word tokenize query
        query_lst = [word for word in word_tokenize(query)]

        return query_lst

    @staticmethod
    def get_entities(lst):
        return [word for word, tag in PageMatcher.stanford_ner.tag(lst) if tag != "O"]

    @staticmethod
    def get_nouns(lst, include_pronouns=True):
        if include_pronouns:
            return [word for word, tag in nltk.tag.pos_tag(lst) if tag.startswith("NN")]
        return [word for word, tag in nltk.tag.pos_tag(lst) if tag.startswith("NN") and not tag.startswith("NNP")]
