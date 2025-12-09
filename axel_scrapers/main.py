import os
import csv
import argparse

def load_start_urls(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def ensure_metadata_dir(base_dir: str = "data") -> str:
    metadata_dir = os.path.join(base_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    return metadata_dir

def run_scraper(
    name: str,
    scraper_cls,
    config_file: str,
    meta_filename: str,
    base_output_dir: str = "data",
) -> None:
    start_urls = load_start_urls(config_file)
    scraper = scraper_cls(base_output_dir=base_output_dir)
    pdfs = scraper.crawl(start_urls)

    metadata_dir = ensure_metadata_dir(base_output_dir)
    meta_path = os.path.join(metadata_dir, meta_filename)

    fieldnames = ["url", "path", "status", "sha256", "product_name", "source_page"]
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in pdfs:
            writer.writerow(row)

    print(f"[{name}] Saved {len(pdfs)} PDF metadata rows to {meta_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Scrape product carbon footprint PDFs from manufacturers."
    )
    parser.add_argument(
        "vendor",
        choices=["apple", "lenovo", "dell", "hp", "all"],
        help="Which vendor scraper to run."
    )
    args = parser.parse_args()

    jobs = {
        "apple": {
            "import": lambda: __import__("scrapers.apple", fromlist=["AppleScraper"]).AppleScraper,
            "config": "config/apple_start_urls.txt",
            "meta": "apple_pdfs.csv",
        },
        "lenovo": {
            "import": lambda: __import__("scrapers.lenovo", fromlist=["LenovoScraper"]).LenovoScraper,
            "config": "config/lenovo_start_urls.txt",
            "meta": "lenovo_pdfs.csv",
        },
        "dell": {
            "import": lambda: __import__("scrapers.dell", fromlist=["DellScraper"]).DellScraper,
            "config": "config/dell_start_urls.txt",
            "meta": "dell_pdfs.csv",
        },
        "hp": {
            "import": lambda: __import__("scrapers.hp", fromlist=["HPScraper"]).HPScraper,
            "config": "config/hp_start_urls.txt",
            "meta": "hp_pdfs.csv",
        },
    }

    if args.vendor == "all":
        for name, job in jobs.items():
            print(f"\nRUNNING {name.upper()} SCRAPER")
            scraper_cls = job["import"]()
            run_scraper(
                name=name,
                scraper_cls=scraper_cls,
                config_file=job["config"],
                meta_filename=job["meta"],
            )
    else:
        job = jobs[args.vendor]
        scraper_cls = job["import"]()
        run_scraper(
            name=args.vendor,
            scraper_cls=scraper_cls,
            config_file=job["config"],
            meta_filename=job["meta"],
        )

if __name__ == "__main__":
    main()