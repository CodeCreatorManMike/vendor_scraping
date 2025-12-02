from typing import List, Dict
from bs4 import BeautifulSoup

from .base import BaseScraper

class LenovoScraper(BaseScraper):
    def __init__(self, base_output_dir: str = "data"):
        super().__init__("lenovo", base_output_dir)

    def extract_pdf_links_from_page(self, url: str) -> List[dict]:
        # find all <a> tags, which point to .pdf
        # extract product name from the hyperlinked text
        # download PDF's
        # return metadata list

        soup = self.get_soup(url)
        pdf_metadata: List[dict] = []

        # general find all <a href="...pdf">
        for a in soup.find_all("a", href=True):
            href = a["href"]
            abs_url = self.make_absolute(url, href)

            if not self.looks_like_pdf_url(abs_url):
                continue

            # filter for only PCF / ECO Declaration sheets
            # (lenovo uses 'eco-declaration' in the path and 'pcf' in filenames)
            link_text = (a.get_text(" ", strip=True) or "").lower()
            url_l = abs_url.lower()

            if (
                "eco-declaration" not in url_l
                and "pcf" not in url_l
                and "product carbon footprint" not in link_text
                and "eco declaration" not in link_text
            ):
                continue

            # product name heuristics:
            # usually the visible link text is something like "ThinkPad E15 Gen 2 PCF"
            product_name = a.get_text(" ", strip=True) or None

            meta = self.save_pdf(abs_url, suggested_name=product_name)
            meta["product_name"] = product_name
            meta["source_page"] = url
            pdf_metadata.append(meta)

        return pdf_metadata

    def crawl(self, start_urls: List[str]) -> List[dict]:
        """
        For Lenovo, loop over the ECO/PCF listing or resource pages,
        extract PDF links from each, download PDFs, return aggregated metadata.
        """
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

        return all_meta
