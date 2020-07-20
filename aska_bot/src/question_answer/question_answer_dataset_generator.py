import json
import pandas as pd
from collections import OrderedDict


def generate_question_answer_dataset(data):
    data_dict = OrderedDict()
    i = 0

    # iterate through each topic
    for index, data_subset in enumerate(data["data"]):
        if index % 25 == 0:
            print("Extracting subset [{}] / [{}]".format(index, len(data["data"])))

        # iterate through each paragraph
        for paragraph in data_subset["paragraphs"]:
            context = paragraph["context"]

            # iterate though each question
            for qas in paragraph["qas"]:
                question = qas["question"]
                is_impossible = qas["is_impossible"]
                answer = None
                answer_bounds = None
                if not is_impossible and len(qas["answers"]) == 1:
                    answer = qas["answers"][0]["text"]
                    answer_start = qas["answers"][0]["answer_start"]
                    answer_bounds = (answer_start, answer_start + len(answer))

                # update data_dict
                data_dict[i] = {"question": question,
                                "answer": answer,
                                "answer_bounds": answer_bounds,
                                "is_impossible": is_impossible,
                                "context": context}
                i += 1

    return pd.DataFrame.from_dict(data_dict, orient="index")


if __name__ == "__main__":
    # read json
    json_path = "../../../data/SQUAD_v2.0/train.json"
    with open(json_path) as f:
        data = json.load(f)

    # convert to dataframe
    df = generate_question_answer_dataset(data)

    # write to csv
    csv_path = "../../../data/question_answer_dataset/question_answer_dataset.csv"
    df.to_csv(csv_path, index=False)
