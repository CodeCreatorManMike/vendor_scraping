import os
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

START_URL = "https://www.dell.com/en-us/lp/dt/product-carbon-footprints"
OUTPUT_DIR = Path("PDFs")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PCF-Scraper/1.0; +https://example.com)"
}


def get_page_html(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_pdf_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    pdf_urls = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if ".pdf" not in href.lower():
            continue

        # Normalise to absolute URL
        full_url = urljoin(base_url, href)
        pdf_urls.add(full_url)

    return sorted(pdf_urls)


def filename_from_url(url: str) -> str:
    """
    Take the last segment of the path as the filename.
    Falls back to something sane if the URL ends with a slash.
    """
    path = urlsplit(url).path
    name = os.path.basename(path)
    if not name:
        # No basename – give it something generic
        name = "download.pdf"

    return name


def download_pdf(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = filename_from_url(url)
    dest = output_dir / filename

    # If you want to overwrite existing files, remove this guard.
    if dest.exists():
        print(f"[SKIP] {dest} already exists")
        return dest

    print(f"[DL]   {url} -> {dest}")

    with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)

    return dest


def main():
    print(f"Fetching landing page: {START_URL}")
    html = get_page_html(START_URL)

    pdf_links = extract_pdf_links(html, START_URL)
    print(f"Found {len(pdf_links)} PDF links")

    if not pdf_links:
        print("No PDFs found — check if the page structure has changed.")
        return

    for url in pdf_links:
        try:
            download_pdf(url, OUTPUT_DIR)
        except Exception as e:
            print(f"[ERR] Failed to download {url}: {e}")


if __name__ == "__main__":
    main()
 
