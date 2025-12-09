from typing import List, Dict
from bs4 import BeautifulSoup
import time
import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from .base import BaseScraper


class LenovoScraper(BaseScraper):
    def __init__(self, base_output_dir: str = "data"):
        super().__init__("lenovo", base_output_dir)
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        )

        self.driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=chrome_options
        )

    def extract_pdf_links_from_page(self, url: str) -> List[dict]:
        pdf_metadata: List[dict] = []
        try:
            self.driver.get(url)
            time.sleep(5)  # Let JS load
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
        except Exception as e:
            print(f"[lenovo] Error rendering {url}: {e}")
            return []

        for a in soup.find_all("a", href=True):
            href = a["href"]
            abs_url = self.make_absolute(url, href)

            # must be a PDF
            if not self.looks_like_pdf_url(abs_url):
                continue

            url_l = abs_url.lower()

            # ---------- STRICT LENOVO PCF FILTER ----------
            # Only keep Lenovo PCF sheets, which:
            #   - live under /regulatory/eco-declaration/
            #   - have filenames starting with "pcf-"
            #     e.g. .../regulatory/eco-declaration/pcf-lenovo-300e-chromebook.pdf
            if "lenovo.com" not in url_l:
                continue
            if "eco-declaration" not in url_l:
                continue

            # Strip querystring then take just the filename
            filename = os.path.basename(url_l.split("?", 1)[0])
            if not filename.startswith("pcf-"):
                continue
            # ------------------------------------------------

            # product name heuristics:
            product_name = a.get_text(" ", strip=True) or None

            meta = self.save_pdf(abs_url, suggested_name=product_name)
            meta["product_name"] = product_name
            meta["source_page"] = url
            pdf_metadata.append(meta)

        return pdf_metadata

    def crawl(self, start_urls: List[str]) -> List[dict]:
        all_meta: List[dict] = []
        seen_urls = set()

        for url in start_urls:
            print(f"\n[lenovo] Crawling start URL: {url}")
            page_meta = self.extract_pdf_links_from_page(url)

            for m in page_meta:
                u = m["url"]
                if u in seen_urls:
                    continue
                seen_urls.add(u)
                all_meta.append(m)

        self.driver.quit()
        return all_meta
