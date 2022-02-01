from typing import Union

from cached_request import CachedRequest


class NasdaqPublic:

    def __init__(self, caching: Union[bool, str] = True):
        self.web = CachedRequest(
            "nasdaq",
            caching=caching,
            headers={
                "user-agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0",
                "Referer":  "https://www.nasdaq.com/",
                "origin":  "https://www.nasdaq.com/",
                "accept": "Accept: application/json, text/plain, */*",
            },
        )

    def search(self, query: str):
        url = f"https://api.nasdaq.com/api/autocomplete/slookup/10"
        return self.web.request(url, params={"search": query})

    def company_profile(self, symbol: str) -> dict:
        url = f"https://api.nasdaq.com/api/company/{symbol}/company-profile"
        return self.web.request(url, json=True)

    def institutional_holdings(
            self,
            id: Union[int, str],
            limit: int = 100,
            type: str = "TOTAL",
    ):
        sort_column = "marketValue" if isinstance(id, str) else "value"
        url = f"https://api.nasdaq.com/api/company/{id}/institutional-holdings" \
              f"?limit={limit}&type={type}&sortColumn={sort_column}&sortOrder=DESC"

        return self.web.request(url, json=True)
