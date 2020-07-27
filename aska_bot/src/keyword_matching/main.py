import os

import torch
from transformers import BertTokenizer

from aska_bot.src.keyword_matching.model import BertKeywordModel

if __name__ == "__main__":
    # model paths
    model_name = "bert-base-uncased"
    version = "v2.0"
    model_path = "../../models/keyword_matcher_{}_{}.pt".format(model_name, version)

    # get tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    if os.path.exists(model_path):
        # load model
        model = BertKeywordModel(model_name)
        model.load_state_dict(torch.load(model_path))

        while True:
            # ask for question
            question = input("Please enter a question: \n")

            # get tokens
            tokens = tokenizer(question)

            # get prediction
            logits = model(torch.tensor([tokens["input_ids"]]),
                           attention_mask=torch.tensor([tokens["attention_mask"]]))
            prediction = torch.argmax(logits, dim=2).tolist()

            # print output
            output_index = [i for i, x in enumerate(prediction[0]) if x == 1]
            output = [tokenizer.convert_ids_to_tokens(tokens["input_ids"][i]) for i in output_index]
            print(output)
    else:
        print("No model found at {}".format(model_path))