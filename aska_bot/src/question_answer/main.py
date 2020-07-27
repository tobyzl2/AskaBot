import torch
from transformers import BertForQuestionAnswering, BertTokenizer

if __name__ == "__main__":
    # model name
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

    # define tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    qna_model = BertForQuestionAnswering.from_pretrained(model_name)

    while True:
        # ask for context and question
        context = input("Please provide a context.\n")
        question = input("What is your question?\n")

        # get tokens
        tokens = tokenizer(question, context)
        input_ids = tokens["input_ids"]
        token_type_ids = tokens["token_type_ids"]

        # get scores
        start_scores, end_scores = qna_model(
            torch.tensor([input_ids]),
            token_type_ids=torch.tensor([token_type_ids])
        )

        # get start and end ids
        start_id = torch.argmax(start_scores)
        end_id = torch.argmax(end_scores)

        # print answer
        answer_ids = input_ids[start_id: end_id + 1]
        answer = tokenizer.decode(answer_ids)
        print("The answer is: {}".format(answer))
