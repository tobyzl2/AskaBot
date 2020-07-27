from aska_bot.src.page_matching.page_matcher import PageMatcher

if __name__ == "__main__":
    query = "What religions and idea of thought is heresy cited as being used frequently in?"
    n_searches = 10

    # query list
    query_lst = PageMatcher.process_query(query)
    print("Query List: {}".format(query_lst))

    # get chunks
    chunks = PageMatcher.get_chunks(query_lst)
    print("Chunks: {}".format(chunks))

    # get relevant_searches
    relevant_searches = PageMatcher.get_relevant_searches(query_lst, chunks, n_searches)
    print("Relevant Searches: {}".format(relevant_searches))
