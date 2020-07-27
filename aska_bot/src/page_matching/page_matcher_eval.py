import pandas as pd

DATASET_PATH = "../../data/page_dataset/page_dataset_1.csv"

from .page_matcher import PageMatcher


def get_acc(df, n_searches=10):
    acc = 0
    for index, row in df.iterrows():
        if index % 25 == 0:
            print("Evaluating: [{}]/[{}]".format(index, len(df)))
        question = row["question"]
        label = row["label"]
        query_lst = PageMatcher.process_query(question)
        chunks = PageMatcher.get_chunks(query_lst)
        relevant_searches = PageMatcher.get_relevant_searches(query_lst, chunks, n_searches)
        if label in relevant_searches:
            acc += 1
        else:
            print("Failed to Evaluate")
            print("Question: {}".format(question))
            print("Label: {}".format(label))
            print("Chunks: {}".format(chunks))
            print("Searches: {}".format(relevant_searches))
    return acc / len(df)


if __name__ == "__main__":
    page_df = pd.read_csv(DATASET_PATH)
    print(get_acc(page_df))
