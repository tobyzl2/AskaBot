import json
import pandas as pd

from .wikipedia_api import WikipediaAPI


def build_page_dataset(data, max_questions=50):
    # TODO: use title to generate labels instead of context
    res = pd.DataFrame(columns=["question", "label"])
    for i, data_subset in enumerate(data["data"]):
        if i % 50 == 0:
            print("Extracting subset: [{}]/[{}]".format(i, len(data["data"])))
        n_questions = 0
        for paragraph in data_subset["paragraphs"]:
            context = paragraph["context"]
            label = WikipediaAPI.find_titles_by_text(context[:300], limit=1)
            if len(label["query"]["search"]) > 0:
                label = label["query"]["search"][0]["title"]
            else:
                continue
            for question in paragraph["qas"]:
                if not question["is_impossible"]:
                    res.loc[len(res)] = {"question": question["question"], "label": label}
                    n_questions += 1
                if n_questions >= max_questions:
                    break
            if n_questions >= max_questions:
                break
    return res


if __name__ == "__main__":
    json_path = "../../data/SQUAD_v2.0/train.json"
    with open(json_path) as f:
        data = json.load(f)
    page_dataset = build_page_dataset(data, max_questions=1)
    csv_path = "../../data/page_dataset/page_dataset_1.csv"
    page_dataset.to_csv(csv_path, index=False)
