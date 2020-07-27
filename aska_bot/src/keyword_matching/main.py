from aska_bot.src.keyword_matching.api import KeywordMatcher

if __name__ == "__main__":
    # model properties
    model_name = "bert-base-uncased"
    version = "v2.0"

    keyword_matcher = KeywordMatcher(model_name, version)

    while True:
        # ask for question
        question = input("Please enter a question: \n")
        print(keyword_matcher.match(question))
