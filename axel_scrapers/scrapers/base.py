import os
import time
import hashlib
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from slugify import slugify

class BaseScraper:
    # basic initialization / OS directory creation
    def __init__(self, name: str, base_output_dir: str = "data"):
        self.name = name.lower()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"AZEL-Scraper/{self.name} (+michael.jones@viadex.com)"
        })
        self.base_output_dir = base_output_dir
        self.pdf_dir = os.path.join(base_output_dir, "pdfs", self.name)
        os.makedirs(self.pdf_dir, exist_ok=True)


    # HTTP helpers
    def fetch(self, url: str, *, timeout: int = 20) -> requests.Response:
        # Get request / basic retry function
        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=timeout)
                if resp.status_code == 200:
                    return resp
                else:
                    print(f"\n[{self.name}] Error fetching {url}: status {resp.status_code}")
            except requests.RequestException as e:
                print(f"\n[{self.name}] Error fetching {url}: {e}")
            if attempt == 2:
                break
            time.sleep(3)

        raise RuntimeError(f"\nFailed to search {url}")
    
    def get_soup(self, url: str) -> BeautifulSoup:
        resp = self.fetch(url)
        return BeautifulSoup(resp.text, "html.parser")
    

    # URL helpers
    def make_absolute(self, base_url: str, link: str) -> str:
        return urljoin(base_url, link)
    
    def looks_like_pdf_url(self, url: str) -> bool:
        # basic check // Note: need to enrich with keyword filter
        parsed = urlparse(url)
        return parsed.path.lower().endswith(".pdf")
    
    # PDF saving
    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    
    def save_pdf(self, url: str, suggested_name: str | None = None) -> dict:
        # download and save PDF to disk
        # this returns a metadata dict: {url, path, sh256, status}
        print(f"\n[{self.name}] Downloading PDF: {url}")
        try:
            resp = self.fetch(url)
            if resp.status_code != 200:
                return {
                    "url": url,
                    "path": None,
                    "status": resp.status_code,
                    "sha256": None,
                }
            
            content = resp.content
            sha = hashlib.sha256(content).hexdigest()

            # build filename
            if suggested_name:
                base = slugify(suggested_name)[:80]
            else:
                base = self._hash(url)

            filename = f"{base}_{sha[:8]}.pdf"
            filepath = os.path.join(self.pdf_dir, filename)

            # de duplication if file already exists
            if not os.path.exists(filepath):
                with open(filepath, "wb") as f:
                    f.write(content)

            return {
                "url": url,
                "path": filepath,
                "status": 200,
                "sha256": sha,
            }
        
        except Exception as e:
            print(f"\n[{self.name}] Failed to download {url}: {e}")
            return {
                "url": url,
                "path": None,
                "status": "error",
                "sha256": None,
            }
        
    # implementing per manufacturer
    def crawl(self, start_urls: list[str]) -> list[dict]:
        """
        Main entrypoint. Implement in subclass:
        Given start URLs, find all relevant PDF links.
        Should return list of PDF metadata dicts.
        """
        raise NotImplementedError
