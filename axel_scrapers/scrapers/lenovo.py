import json
import time
from typing import List, Dict


import requests
from bs4 import BeautifulSoup


from .base import BaseScraper




class HPScraper(BaseScraper):
def __init__(self, base_output_dir: str = "data"):
super().__init__("hp", base_output_dir)
self.library_url = (
"https://h20195.www2.hp.com/v2/library.aspx"
"?doctype=95&footer=95&filter_doctype=no&showregionfacet=yes"
"&filter_country=no&cc=us&lc=en&filter_oid=no&filter_prodtype=rw"
"&prodtype=ij&showproductcompatibility=yes&showregion=yes"
"&showreglangcol=yes&showdescription=yes"
"#doctype-95&sortorder-popular&teasers-off&isRetired-false"
"&isRHParentNode-false&titleCheck-false"
)
self.loadmore_url = "https://h20195.www2.hp.com/v2/Library.aspx/LoadMore"
self.headers = {
"User-Agent": (
"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
"AppleWebKit/537.36 (KHTML, like Gecko) "
"Chrome/127.0.0.0 Safari/537.36"
),
"Accept": "application/json, text/javascript, */*; q=0.01",
"Accept-Language": "en-US,en;q=0.9",
"Origin": "https://h20195.www2.hp.com",
"Referer": self.library_url,
"Content-Type": "application/json",
"X-Requested-With": "XMLHttpRequest",
"Cache-Control": "no-cache",
}


def crawl(self, start_urls: List[str]) -> List[Dict]:
session = requests.Session()
session.verify = False # disable SSL verification
session.headers.update(self.headers)


# Prime session with initial GET
session.get(
self.library_url,
headers={k: v for k, v in self.headers.items() if k not in ("Content-Type", "X-Requested-With")},
timeout=60,
)


print(f"[hp] Crawling HP library via AJAX (LoadMore endpoint)")
request_id = str(int(time.time()))
click = 1
grid = 220
step = 200
last_current = 0


seen_urls = set()
all_meta: List[Dict] = []


while True:
body = {
"test": "",
"hdnOffset": str(click),
"sort": "popular",
"facets": "document_type#95",
"uniqueRequestId": request_id,
"titleSearch": "false",
"cc": "us",
"lc": "en",
"totalcount": "0",
"gridCount": str(grid),
}


try:
resp = session.post(
self.loadmore_url,
headers=self.headers,
data=json.dumps(body),
return all_meta
