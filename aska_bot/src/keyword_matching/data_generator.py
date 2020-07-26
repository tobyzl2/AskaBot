from collections import OrderedDict

import json
import re
import pandas as pd
from nltk.corpus import stopwords

from aska_bot.src.page_matcher.wikipedia_api import WikipediaAPI

SPLIT_REGEX = r"[_\-\–\s]"


def read_data(datapath):
    with open(datapath) as f:
        data = json.load(f)
    return data


def get_target_keywords(title, stop_words):
    target_keywords = WikipediaAPI.find_titles_by_query(" ".join(title), results=1)
    if not target_keywords:
        return None

    target_keywords = re.split(SPLIT_REGEX, target_keywords)
    return [word for word in target_keywords if word.lower() not in stop_words]


def build_keyword_dataset(datapath):
    stop_words = set(stopwords.words('english'))
    res = OrderedDict()
    i = 0

    # read data
    data = read_data(datapath)

    for data_index, data_subset in enumerate(data["data"]):
        # verbose
        if data_index % 50 == 0:
            print("Generating Iteration: [{}]/[{}]".format(data_index, len(data["data"])))

        # get target keywords
        title = re.split(SPLIT_REGEX, data_subset["title"])
        target_keywords = get_target_keywords(title, stop_words)
        if not target_keywords:
            continue

        # iterate through each question
        for paragraph in data_subset["paragraphs"]:
            for question in paragraph["qas"]:
                if not question["is_impossible"]:
                    # get question and keywords
                    question_text = question["question"].replace("'s", " 's")
                    keywords = [word for word in question_text.split(" ") if word in target_keywords]

                    # add to result
                    if keywords:
                        res[i] = {"question": question_text, "keywords": keywords, "title": data_subset["title"]}
                        i += 1

    return pd.DataFrame().from_dict(res, orient="index")


if __name__ == "__main__":
    json_path = "../../../data/SQUAD_v2.0/train.json"
    df = build_keyword_dataset(json_path)

    out_path = "../../../data/keyword_dataset/keyword_dataset.csv"
    df.to_csv(out_path, index=False)
