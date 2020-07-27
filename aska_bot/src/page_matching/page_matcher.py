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

# instantiate stanford ner
stanford_ner = StanfordNERTagger("../../models/english.conll.4class.distsim.crf.ser.gz")


class PageMatcher:
    @staticmethod
    def process_query(query, remove_possessive=True,
                      remove_punctuation=True,
                      remove_stopwords=True):
        """
        Performs string processing such as removing possessive, punctuation, and stopword.
        """
        if remove_possessive:
            # remove possessive
            query = query.replace("\'s", "")
            query = query.replace("s\'", "")

        if remove_punctuation:
            # remove punctuation
            query = query.translate(str.maketrans("", "", string.punctuation))

        if remove_stopwords:
            # remove stopwords
            eng_stopwords = stopwords.words('english')
            for word in eng_stopwords:
                # TODO: handle first word
                if re.search(r" {} ".format(word), query):
                    query = query.replace(r"{} ".format(word), "")

        # word tokenize query
        query_lst = [word for word in word_tokenize(query)]

        return query_lst

    @staticmethod
    def get_consec_ner(lst):
        """
        Groups nouns based on consecutive named-entities and consecutive non-named-entities.
        """
        consec_ner = []
        temp = []
        is_entity = False
        ner_tags = stanford_ner.tag(lst)

        for word, tag in ner_tags:
            if (tag != "O" and is_entity) or (tag == "O" and not is_entity):
                temp.append(word)
            else:
                if temp:
                    consec_ner.append(temp)
                temp = [word]
                is_entity = not is_entity

        # add remaining words in temp
        if temp:
            consec_ner.append(temp)

        return consec_ner

    @staticmethod
    def get_consec_combinations(lst):
        """
        Gets consecutive combinations in a list where each combination has two or more elements.
        """
        consec_combinations = []
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                consec_combinations.append(lst[i:j + 1])

        return consec_combinations

    @staticmethod
    def chunk_nouns(nouns):
        return PageMatcher.chunk_nouns_recursive(nouns, list())

    @staticmethod
    def chunk_nouns_recursive(rem_nouns, chunks):
        if len(rem_nouns) == 0:
            return chunks

        if len(rem_nouns) == 1:
            chunks.append(rem_nouns[0])
            return chunks

        all_combinations = PageMatcher.get_consec_combinations(rem_nouns)

        results = []
        for combination in all_combinations:
            combination_str = " ".join(combination)
            new_rem_nouns = rem_nouns.copy()
            new_chunks = chunks.copy()
            if WikipediaAPI.has_near_match(combination_str):
                for noun in combination:
                    new_rem_nouns.remove(noun)
                new_chunks.append(combination_str)
                result = PageMatcher.chunk_nouns_recursive(new_rem_nouns, new_chunks)
                results.append(result)

        if results:
            max_result = min(results, key=len)
            return max_result
        else:
            chunks.extend(rem_nouns)
            return chunks

    @staticmethod
    def get_chunks(query_lst):
        res = []
        for consecutive_ner in PageMatcher.get_consec_ner(query_lst):
            if len(consecutive_ner) > 1:
                res.extend(PageMatcher.chunk_nouns(consecutive_ner))
            else:
                res.append(consecutive_ner[0])
        tagged = nltk.pos_tag(res)
        res = [tag[0] for tag in tagged if tag[1][0] == "N"]
        return res

    @staticmethod
    def get_relevant_searches(query_lst, chunks, n_searches):
        if n_searches > len(chunks) + 1:
            relevant_searches = []
            n_searches_per_chunk = n_searches // (len(chunks) + 1)

            for chunk in chunks:
                n_chunk_searches = 0
                searches = WikipediaAPI.find_titles_by_query(chunk, n_searches)
                for search in searches:
                    # find searches not already found
                    if n_chunk_searches >= n_searches_per_chunk:
                        break
                    if search not in relevant_searches:
                        relevant_searches.append(search)
                        n_chunk_searches += 1

            searches = WikipediaAPI.find_titles_by_query(" ".join(query_lst), n_searches)
            for search in searches:
                if len(relevant_searches) >= n_searches:
                    break
                if search not in relevant_searches:
                    relevant_searches.append(search)
        else:
            relevant_searches = WikipediaAPI.find_titles_by_query(" ".join(query_lst), n_searches)

        return relevant_searches
