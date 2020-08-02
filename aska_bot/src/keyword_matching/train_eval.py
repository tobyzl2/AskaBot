import torch
import os

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertTokenizer

from aska_bot.src.keyword_matching.keyword_dataset import get_input, get_dataloaders
from aska_bot.src.keyword_matching.model import BertKeywordModel


def train(model, tokenizer, keyword_train_dataloader, epochs=5, lr=1e-4, out_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, (questions, keywords) in enumerate(keyword_train_dataloader):
            # compute loss
            input_ids, attention_mask, labels = get_input(questions, keywords, tokenizer)
            loss = model(input_ids, attention_mask=attention_mask, labels=labels)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # verbose
            print("Epoch: [{}]/[{}], Iteration: [{}]/[{}], Loss: {}"
                  .format(epoch+1, epochs, i+1, len(keyword_train_dataloader), loss.item()))

    if out_path:
        torch.save(model.state_dict(), out_path)


def evaluate(model, tokenizer, keyword_test_dataloader, metric_dict):
    metrics = {}
    predicted = []
    expected = []

    model.eval()
    for i, (questions, keywords) in enumerate(keyword_test_dataloader):
        # compute logits
        input_ids, attention_mask, label = get_input(questions, keywords, tokenizer)
        logits = model(input_ids, attention_mask=attention_mask)

        # update predicted and expected
        prediction = torch.argmax(logits, dim=2)
        predicted.extend(prediction.flatten().tolist())
        expected.extend(label.flatten().tolist())

        # verbose
        print("Iteration: [{}]/[{}]".format(i+1, len(keyword_test_dataloader)))

    # compute metrics
    for metric_name, metric in metric_dict.items():
        metrics[metric_name] = metric(expected, predicted)

    return metrics


if __name__ == "__main__":
    # dataset paths
    keyword_dataset_path = "../../../data/keyword_dataset/keyword_dataset.csv"
    json_path = "../../../data/SQUAD_v2.0/train.json"

    # model paths
    model_name = "bert-base-cased"
    version = "v2.0"
    model_path = "../../models/keyword_matcher_{}_{}.pt".format(model_name, version)

    # hyperparameters
    random_seed = 42
    test_size = 0.1
    batch_size = 256
    lr = 5.0e-6
    epochs = 5

    # reproducibility
    torch.manual_seed(random_seed)

    # metrics
    metrics_dict = {
        "accuracy": accuracy_score,
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score
    }

    # get dataloaders
    keyword_train_dataloader, keyword_test_dataloader = get_dataloaders(
        keyword_dataset_path, json_path, batch_size, test_size, random_seed
    )

    # get tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    if os.path.exists(model_path):
        # load model and evaluate
        model = BertKeywordModel(model_name)
        model.load_state_dict(torch.load(model_path))

        metrics = evaluate(model, tokenizer, keyword_test_dataloader, metrics_dict)
    else:
        # train model and evaluate
        model = BertKeywordModel(model_name)

        train(model, tokenizer, keyword_train_dataloader, epochs, lr, model_path)
        metrics = evaluate(model, tokenizer, keyword_test_dataloader, metrics_dict)

    # print metrics
    for metric_name, metric_val in metrics.items():
        print("{}: {}".format(metric_name, metric_val))
