import sys
import os
import tempfile
import json as jsonlib
from pathlib import Path
from typing import Union, Optional, List

import requests
import bs4

from cached_request import CachedRequest


class Edgar:

    USER_AGENT = "private investigation (broker442@gmail.com)"

    def __init__(self, caching: Union[bool, str] = True):
        self.web = CachedRequest(
            "sec_edgar",
            caching=caching,
            headers={
                "user-agent": self.USER_AGENT,
                "accept-encoding": "gzip, deflate",
                # This is making problems getting raw json files
                # "host": "www.sec.gov",
            }
        )
        self._company_tickers_exchange: dict = None
        self._company_map = dict()
        self._company_data = dict()

    @property
    def company_tickers_exchange(self) -> dict:
        if self._company_tickers_exchange is None:
            self._company_tickers_exchange = self.web.request(
                "https://www.sec.gov/files/company_tickers_exchange.json",
                json=True,
            )
        return self._company_tickers_exchange

    def get_company(self, query: Union[int, str]) -> Optional["Company"]:
        if query in self._company_map:
            return self._company_map[query]

        company = None
        if isinstance(query, int):
            for cte in self.company_tickers_exchange["data"]:
                if cte[0] == query:
                    company = Company(self, *cte)
                    break
        else:
            query = query.lower()
            for cte in self.company_tickers_exchange["data"]:
                if query in cte[1].lower() or query in cte[2]:
                    company = Company(self, *cte)
                    break
        if company:
            self._company_map[query] = company
        return company

    def get_company_data(self, cik: int) -> dict:
        if cik not in self._company_data:
            self._company_data[cik] = self.web.request(
                f"https://data.sec.gov/submissions/CIK{cik:010}.json",
                json=True,
            )
        return self._company_data[cik]


class Company:

    def __init__(self, edgar: Edgar, cik: int, name: str, ticker: str, exchange: Optional[str] = None):
        self.edgar = edgar
        self.cik = cik
        self.name = name
        self.ticker = ticker
        self.exchange = exchange
        self._filings = None

    def __str__(self):
        return f"{self.cik}/{self.name}"

    @property
    def data(self):
        return self.edgar.get_company_data(self.cik)

    def filings(self, filter: Optional[str] = None) -> List["Filing"]:
        def add_filings(filings_obj):
            for i in range(len(filings_obj["accessionNumber"])):
                form = filings_obj["form"][i]
                if filter is not None and form != filter:
                    continue

                filing_class_name = f"Filing{form}"
                filing_class = globals().get(filing_class_name, Filing)
                filing = filing_class(self)
                for field in Filing.FIELDS:
                    setattr(filing, field, filings_obj[field][i])
                self._filings.append(filing)

        if self._filings is None:
            self._filings = []
            add_filings(self.data["filings"]["recent"])
            if self.data["filings"].get("files"):
                for file in self.data["filings"]["files"]:
                    filing_obj = self.edgar.web.request(
                        f"https://data.sec.gov/submissions/{file['name']}",
                        json=True,
                    )
                    add_filings(filing_obj)

        return self._filings


class Filing:

    FIELDS = [
        "accessionNumber",
        "filingDate",
        "reportDate",
        "acceptanceDateTime",
        "act",
        "form",
        "fileNumber",
        "filmNumber",
        "items",
        "size",
        "isXBRL",
        "isInlineXBRL",
        "primaryDocument",
        "primaryDocDescription",
    ]

    def __init__(self, company: Company):
        self.company = company
        self.accessionNumber: str = None
        self.filingDate: str = None
        self.reportDate: str = None
        self.acceptanceDateTime: str = None
        self.act: str = None
        self.form: str = None
        self.fileNumber: str = None
        self.filmNumber: str = None
        self.items: str = None
        self.size: int = None
        self.isXBRL: int = None
        self.isInlineXBRL: int = None
        self.primaryDocument: str = None
        self.primaryDocDescription: str = None
        self._primary_document = None
        self._xml_document = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.filingDate}/{self.accessionNumber}/{self.form})"

    @property
    def edgar(self) -> Edgar:
        return self.company.edgar

    def to_dict(self) -> dict:
        return {
            key: getattr(self, key)
            for key in self.FIELDS
        }

    def primary_document(self, html: bool = False):
        if not self._primary_document:
            filename = self.primaryDocument
            if not html:
                filename = filename.split("/")[-1]
            url = "https://www.sec.gov/Archives/edgar/data/%s/%s/%s" % (
                self.company.cik,
                self.accessionNumber.replace("-", ""),
                filename,
            )
            self._primary_document = self.edgar.web.request(url).decode("utf-8")
        return self._primary_document


class Filing4(Filing):

    @property
    def soup(self) -> bs4.BeautifulSoup:
        if not hasattr(self, "_soup"):
            self._soup = bs4.BeautifulSoup(self.primary_document(), features="lxml")
        return self._soup

    def element_value(self, name_or_elem: Union[str, bs4.Tag], name: Optional[str] = None) -> Optional[str]:
        if name:
            elem = name_or_elem
        else:
            elem = self.soup
            name = name_or_elem
        elem = elem.find(name.lower())
        if elem:
            value = elem.text
            if value:
                value = value.strip()
            return value

    def reporter(self) -> dict:
        try:
            return {
                key: value
                for key, value in {
                    "cik": int(self.element_value("rptOwnerCik").lstrip("0")),
                    "name": self.element_value("rptOwnerName"),
                    "is_director": self.element_value("isDirector") == "1",
                    "is_officer": self.element_value("isOfficer") == "1",
                    "is_ten_percent": self.element_value("isTenPercentOwner") == "1",
                    "is_other": self.element_value("isOther") == "1",
                    "officer_title": self.element_value("officerTitle"),
                    "other": self.element_value("otherText"),
                }.items()
                if value
            }
        except:
            print(jsonlib.dumps(self.to_dict(), indent=2))
            print(self.primary_document())
            raise

    def issuer(self) -> Company:
        try:
            cik = int(self.soup.find("issuercik").text.lstrip("0"))
        except:
            print(jsonlib.dumps(self.to_dict(), indent=2))
            print(self.primary_document())
            raise
        company = self.edgar.get_company(cik)
        if not company:
            company = Company(
                edgar=self.edgar,
                cik=cik,
                name=self.soup.find("issuername").text,
                ticker=self.soup.find("issuertradingsymbol").text,
            )
            self.edgar._company_map[cik] = company
        return company

    def signature(self) -> str:
        return self.soup.find("signaturename").text

    def transactions(self) -> List[dict]:
        transactions = []
        for tag in self.soup.find_all("nonderivativetransaction"):
            try:
                price = float(self.element_value(tag, "transactionPricePerShare"))
            except ValueError:
                price = None
            transactions.append({
                "date": self.element_value(tag, "transactionDate"),
                "title": self.element_value(tag, "securityTitle"),
                "code": self.element_value(tag, "transactionCode"),
                "shares": float(self.element_value(tag, "transactionShares")),
                "price": price,
                "disposed": self.element_value(tag, "transactionAcquiredDisposedCode") == "D",
                "shares_owned": self.element_value(tag, "sharesOwnedFollowingTransaction"),
                "direct_ownership": self.element_value(tag, "directOrIndirectOwnership") == "D",
                "ownership_nature": self.element_value(tag, "natureOfOwnership"),
            })
        return transactions


