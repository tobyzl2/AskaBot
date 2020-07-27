import logging
import os

import torch
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize

from aska_bot.src.keyword_matching.model import BertKeywordModel


class KeywordMatcher:
    DEFAULT_MODEL_NAME = "bert-base-uncased"
    DEFAULT_VERSION = "v2.0"
    logger = logging.getLogger()

    def __init__(self, model_name=None, version=None):
        model_path = "../../models/keyword_matcher_{}_{}.pt".format(model_name, version)
        if not model_name or not version or not os.path.exists(model_path):
            # model paths
            model_name = KeywordMatcher.DEFAULT_MODEL_NAME
            version = KeywordMatcher.DEFAULT_VERSION
            model_path = "../../models/keyword_matcher_{}_{}.pt".format(model_name, version)

            # log warning
            warning_msg = "Failed to find model, defaulting to model at path {}.".format(model_path)
            KeywordMatcher.logger.warning(warning_msg)

        # set model properties
        self.model_name = model_name
        self.version = version
        self.model_path = model_path

        # load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertKeywordModel(model_name)
        self.model.load_state_dict(torch.load(model_path))

    def match(self, query):
        # get tokens
        tokens = self.tokenizer(query)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # get prediction
        logits = self.model(
            torch.tensor([input_ids]),
            attention_mask=torch.tensor([attention_mask])
        )
        prediction = torch.argmax(logits, dim=2).tolist()

        # return output
        output_ids = [input_ids[i] for i, x in enumerate(prediction[0]) if x == 1]
        return word_tokenize(self.tokenizer.decode(output_ids))
