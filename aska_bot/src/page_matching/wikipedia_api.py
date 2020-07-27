import requests
import wikipedia


class WikipediaAPI:
    url = "https://en.wikipedia.org/w/api.php"

    @staticmethod
    def find_titles_by_text(query, limit=10):
        params = {
            "srsearch": query,
            "srlimit": limit,
            "srwhat": "text",
            "action": "query",
            "format": "json",
            "list": "search",
        }
        res = requests.get(WikipediaAPI.url, params)
        return res.json()

    @staticmethod
    def has_near_match(query):
        params = {
            "srsearch": query,
            "srwhat": "nearmatch",
            "action": "query",
            "format": "json",
            "list": "search",
        }
        res = requests.get(WikipediaAPI.url, params)
        return len(res.json()["query"]["search"]) > 0

    @staticmethod
    def find_titles_by_query(query, results=10):
        if results == 1:
            titles = wikipedia.search(query, results)
            if titles:
                return wikipedia.search(query, results)[0]

        return wikipedia.search(query, results)