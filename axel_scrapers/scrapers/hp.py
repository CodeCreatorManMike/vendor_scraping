from typing import List, Dict
from bs4 import BeautifulSoup

from .base import BaseScraper

class HPScraper(BaseScraper):
    def __init__(self, base_output_dir: str = "data"):
        super().__init__("hp", base_output_dir)

    def extract_pdf_links_from_page(self, url: str) -> List[dict]:
        # find all <a> tags, which point to .pdf
        # extract product name from the hyperlinked text
        # download PDF's
        # return metadata list

        soup = self.get_soup(url)
        pdf_metadata: List[dict] = []

        # general find all <a href="...pdf"> or HP-style GetDocument/GetPDF links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            abs_url = self.make_absolute(url, href)
            url_l = abs_url.lower()

            # hp sometimes serves PDFs via GetDocument.aspx (no .pdf in path)
            is_pdf_like = (
                self.looks_like_pdf_url(abs_url)
                or "getdocument.aspx" in url_l
                or "getpdf.aspx" in url_l
            )
            if not is_pdf_like:
                continue

            link_text = (a.get_text(" ", strip=True) or "").lower()

            # filter for only product carbon footprint reports
            # typical phrases: "product carbon footprint", "carbon footprint report"
            if (
                "product carbon footprint" not in link_text
                and "carbon footprint" not in link_text
                and "pcf" not in link_text
            ):
                # you can relax this if your HP listing page is already PCF-only
                continue

            # product name heuristics:
            # often link text is like "HP EliteBook 845 14 inch G10 - Product Carbon Footprint Report"
            product_name = a.get_text(" ", strip=True) or None

            meta = self.save_pdf(abs_url, suggested_name=product_name)
            meta["product_name"] = product_name
            meta["source_page"] = url
            pdf_metadata.append(meta)

        return pdf_metadata

    def crawl(self, start_urls: List[str]) -> List[dict]:
        """
        For HP, loop over Product Carbon Footprint library / listing pages
        or product environmental pages, extract PDF links from each,
        download PDFs, return aggregated metadata.
        """
        all_meta: List[dict] = []
        seen_urls = set()

        for url in start_urls:
            print(f"\n[hp] Crawling start URL: {url}")
            page_meta = self.extract_pdf_links_from_page(url)

            for m in page_meta:
                u = m["url"]
                if u in seen_urls:
                    continue
                seen_urls.add(u)
                all_meta.append(m)

        return all_meta
