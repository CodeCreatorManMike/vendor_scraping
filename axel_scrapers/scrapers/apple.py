"""
Apple PDF Scraper - Downloads environment reports from Apple's website.
Auto-installs dependencies if missing.
"""
import subprocess
import sys
import importlib.util
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup

def check_and_install_package(import_name, package_name):
    """Check if a package is installed, and install it if missing."""
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        # Package not found, install it
        print(f"Installing missing package: {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--quiet"], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Successfully installed {package_name}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package_name}. Please install it manually: pip install {package_name}")
            sys.exit(1)

# Check and install required packages
required_packages = {
    "bs4": "beautifulsoup4",
    "requests": "requests",
    "lxml": "lxml",
    "pypdf": "pypdf",
    "cryptography": "cryptography"
}

for import_name, package_name in required_packages.items():
    check_and_install_package(import_name, package_name)

# Handle both relative and absolute imports
try:
    from .base import BaseScraper
except ImportError:
    # If relative import fails, try absolute import
    try:
        from base import BaseScraper
    except ImportError:
        # If both fail, try importing from parent directory
        from pathlib import Path
        parent_dir = Path(__file__).parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        from base import BaseScraper


class AppleScraper(BaseScraper):
    def __init__(self, base_output_dir: str = "PDFs"):
        # Initialize base class but we'll override the output_dir
        super().__init__("apple", base_output_dir)
        # Override to save directly to PDFs folder, not PDFs/apple/
        self.output_dir = self.base_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove the apple subdirectory if it was created by base class
        apple_subdir = self.base_output_dir / "apple"
        if apple_subdir.exists() and apple_subdir.is_dir():
            # Only remove if it's empty (to avoid deleting files accidentally)
            try:
                if not any(apple_subdir.iterdir()):
                    apple_subdir.rmdir()
            except (OSError, FileNotFoundError):
                pass  # Directory not empty or already removed

    def extract_pdf_links_from_page(self, url: str) -> List[dict]:
        # find all <a> tags, which point to .pdf
        # extract product name from the hyperlinked text
        # download PDF's
        # return metadata list

        # initializing BeautifulSoup
        soup = self.get_soup(url)
        pdf_metadata: List[dict] = []

        # general find all <a href="...pdf">
        for a in soup.find_all("a", href=True):
            href = a["href"]
            abs_url = self.make_absolute(url, href)

            if not self.looks_like_pdf_url(abs_url):
                continue

            # filter for only environmental reports (check both spellings)
            link_text = (a.get_text(" ", strip=True) or "").lower()
            url_lower = abs_url.lower()
            # Check for either spelling: "environment" (correct) or "enviroment" (typo)
            has_env_keyword = (
                "environment" in url_lower or "enviroment" in url_lower or
                "environment" in link_text or "enviroment" in link_text
            )
            if not has_env_keyword:
                continue  # Skip PDFs that aren't environment reports 

            # product name heuristics:
            product_name = a.get_text(" ", strip=True) or None

            meta = self.save_pdf(abs_url, suggested_name=product_name)
            meta["product_name"] = product_name
            meta["source_page"] = url
            pdf_metadata.append(meta)

        
        return pdf_metadata
    
    def crawl(self, start_urls: List[str]) -> List[dict]:
        """
        For Apple, just loop over the start URLs (environment report index pages),
        extract PDF links from each, download PDFs, return aggregated metadata.
        """
        all_meta: List[dict] = []
        seen_urls = set()

        for url in start_urls:
            print(f"\n[apple] Crawling start URL: {url}")
            page_meta = self.extract_pdf_links_from_page(url)

            for m in page_meta:
                u = m["url"]
                if u in seen_urls:
                    continue
                seen_urls.add(u)
                all_meta.append(m)


        return all_meta

    def fix_split_numbers(self, text: str) -> str:
        """
        Fix numbers that were split during PDF extraction.
        Example: "1 3-inch" -> "13-inch", "1 6-inch" -> "16-inch"
        
        Args:
            text: Text that may contain split numbers
            
        Returns:
            Text with split numbers fixed
        """
        # Pattern: single digit, space, single digit followed by "-inch" or at word boundary
        # This fixes cases like "1 3-inch" -> "13-inch"
        text = re.sub(r'(\d)\s+(\d)(-inch)', r'\1\2\3', text)
        # Also fix cases where numbers are split in general (like "1 3" -> "13" when followed by common patterns)
        text = re.sub(r'(\d)\s+(\d)(?=\s|$|[^\d])', r'\1\2', text)
        return text

    def extract_title_and_model_from_pdf(self, pdf_path: Path) -> Optional[Tuple[str, str]]:
        """
        Extract title and model from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (title, model) if found, None otherwise
        """
        try:
            import pypdf
            
            with open(pdf_path, "rb") as file:
                try:
                    pdf_reader = pypdf.PdfReader(file)
                except Exception as e:
                    # Try to handle encrypted PDFs or other issues
                    if "cryptography" in str(e).lower() or "encrypted" in str(e).lower():
                        print(f"  [WARN] PDF {pdf_path.name} appears to be encrypted or requires cryptography library")
                    return None
                
                # Extract text from first few pages (usually title is on first page)
                text = ""
                max_pages = min(10, len(pdf_reader.pages))  # Check first 10 pages for better coverage
                for i in range(max_pages):
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                # If no text extracted, the PDF might be image-based or corrupted
                if not text.strip():
                    return None
                
                lines = [line.strip() for line in text.split("\n") if line.strip()]  # Remove empty lines
                title = None
                model = None
                
                # Product keywords to look for
                product_keywords = ["macbook", "iphone", "ipad", "apple watch", "mac", "imac", 
                                  "mac pro", "mac mini", "airpods", "led cinema display", 
                                  "cinema display", "studio display", "pro display", "vision pro",
                                  "apple tv", "homepod", "magic", "keyboard", "mouse", "trackpad"]
                
                # Strategy 1: Look for "Product Environmental Report" pattern followed by product name
                for i, line in enumerate(lines[:60]):
                    line_lower = line.lower()
                    
                    if "product environmental report" in line_lower:
                        # Look at next few lines for the product name
                        for j in range(i + 1, min(i + 8, len(lines))):
                            next_line = lines[j]
                            next_line_lower = next_line.lower()
                            
                            # Skip common headers/patterns and long sentences
                            skip_patterns = ["report", "environmental", "date", "introduced", "progress", 
                                           "toward", "our", "goal", "responsible", "packaging", "continue",
                                           "innovate", "machine learning", "digitize", "life cycle"]
                            
                            # More strict: must be short (likely a title), contain product keyword, not be a sentence
                            is_likely_title = (
                                len(next_line) <= 80 and  # Titles are usually shorter
                                len(next_line.split()) <= 10 and  # Not a long sentence
                                not any(char in next_line for char in [".", "!", "?"]) or len(next_line) <= 30  # Not a sentence (unless very short)
                            )
                            
                            if (next_line and 
                                is_likely_title and
                                not any(pattern in next_line_lower for pattern in skip_patterns) and
                                any(keyword in next_line_lower for keyword in product_keywords)):
                                title = next_line
                                break
                        if title:
                            break
                
                # Strategy 2: Look for product titles as major headings (often first or early in document)
                if not title:
                    for i, line in enumerate(lines[:40]):
                        line_lower = line.lower()
                        
                        # Check if line contains product keywords and looks like a title
                        if any(keyword in line_lower for keyword in product_keywords):
                            # Clean up the line
                            title_candidate = line
                            # Remove common suffixes and prefixes
                            title_candidate = re.sub(r'^#+\s*', '', title_candidate)  # Remove markdown headers
                            title_candidate = re.sub(r'\s+Environmental\s+(Status\s+)?Report.*$', '', title_candidate, flags=re.IGNORECASE)
                            title_candidate = re.sub(r'\s+Environmental\s+Report.*$', '', title_candidate, flags=re.IGNORECASE)
                            title_candidate = re.sub(r'^Product\s+Environmental\s+Report\s*', '', title_candidate, flags=re.IGNORECASE)
                            title_candidate = title_candidate.strip()
                            
                            # More strict checks: must be title-like, not a sentence
                            is_likely_title = (
                                len(title_candidate) <= 80 and
                                len(title_candidate.split()) <= 10 and
                                (not any(char in title_candidate for char in [".", "!", "?"]) or len(title_candidate) <= 30)
                            )
                            
                            # Skip if it looks like a sentence with common words
                            skip_sentence_words = ["continue", "innovate", "using", "machine", "learning", "digitize", 
                                                  "data", "from", "chemical", "alloy", "metallic", "material", "homogeneous"]
                            has_sentence_words = any(word in title_candidate.lower() for word in skip_sentence_words)
                            
                            # Make sure it's a reasonable title (not too short, not too long, contains product keyword)
                            if (title_candidate and 
                                5 <= len(title_candidate) <= 80 and
                                is_likely_title and
                                not has_sentence_words and
                                any(keyword in title_candidate.lower() for keyword in product_keywords)):
                                # Additional check: make sure it's not just a generic phrase
                                if not all(x in title_candidate.lower() for x in ["product", "environmental"]):
                                    title = title_candidate
                                    break
                
                # Strategy 3: Look for product names that might be on the same line as "Product Environmental Report"
                if not title:
                    for line in lines[:40]:
                        line_lower = line.lower()
                        if "product environmental report" in line_lower:
                            # Try to extract product name from the same line or nearby
                            # Pattern: "Product Environmental Report [Product Name]" or "[Product Name] Product Environmental Report"
                            match = re.search(r'Product\s+Environmental\s+Report\s+(.+?)(?:\s+Date|\s*$)', line, re.IGNORECASE)
                            if match:
                                potential_title = match.group(1).strip()
                                if (potential_title and 
                                    5 <= len(potential_title) <= 120 and
                                    any(keyword in potential_title.lower() for keyword in product_keywords)):
                                    title = potential_title
                                    break
                            
                            # Try reverse pattern
                            match = re.search(r'(.+?)\s+Product\s+Environmental\s+Report', line, re.IGNORECASE)
                            if match:
                                potential_title = match.group(1).strip()
                                if (potential_title and 
                                    5 <= len(potential_title) <= 120 and
                                    any(keyword in potential_title.lower() for keyword in product_keywords)):
                                    title = potential_title
                                    break
                
                # Strategy 4: Look for any prominent line with product keywords in first 60 lines
                if not title:
                    for i, line in enumerate(lines[:60]):
                        line_lower = line.lower()
                        # Look for lines that contain product keywords and are reasonably formatted
                        if any(keyword in line_lower for keyword in product_keywords):
                            # Skip if it's clearly not a title (contains too many common words)
                            common_words = ["the", "and", "or", "with", "for", "from", "this", "that", "these", "those"]
                            word_count = len([w for w in line.split() if w.lower() not in common_words])
                            
                            # Skip sentence-like patterns
                            skip_sentence_words = ["continue", "innovate", "using", "machine", "learning", "digitize", 
                                                  "data", "from", "chemical", "alloy", "metallic", "material", "homogeneous",
                                                  "life cycle", "we're", "we are"]
                            has_sentence_words = any(word in line_lower for word in skip_sentence_words)
                            
                            # Must be title-like (short, not a sentence)
                            is_likely_title = (
                                len(line) <= 80 and
                                len(line.split()) <= 10 and
                                (not any(char in line for char in [".", "!", "?"]) or len(line) <= 30)
                            )
                            
                            if (5 <= len(line) <= 80 and 
                                word_count >= 2 and  # At least 2 meaningful words
                                is_likely_title and
                                not has_sentence_words):
                                title_candidate = line.strip()
                                # Clean up
                                title_candidate = re.sub(r'^#+\s*', '', title_candidate)
                                title_candidate = re.sub(r'\s+Environmental\s+.*$', '', title_candidate, flags=re.IGNORECASE)
                                # Remove date patterns if they're at the end
                                title_candidate = re.sub(r'\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', '', title_candidate)
                                if title_candidate and len(title_candidate) >= 5:
                                    title = title_candidate
                                    break
                
                # Strategy 5: Look for product names that might be split across lines or formatted differently
                if not title:
                    # Try to find product names by looking for common patterns
                    # Look for lines that start with product keywords
                    for i, line in enumerate(lines[:50]):
                        line_stripped = line.strip()
                        line_lower = line_stripped.lower()
                        
                        # Check if line starts with a product keyword
                        for keyword in product_keywords:
                            if line_lower.startswith(keyword) or re.match(rf'^[#\s]*{re.escape(keyword)}', line_lower):
                                # This might be a title
                                title_candidate = line_stripped
                                # Clean up
                                title_candidate = re.sub(r'^#+\s*', '', title_candidate)
                                title_candidate = re.sub(r'\s+Environmental\s+.*$', '', title_candidate, flags=re.IGNORECASE)
                                title_candidate = re.sub(r'\s+Product\s+Environmental\s+Report.*$', '', title_candidate, flags=re.IGNORECASE)
                                
                                if (title_candidate and 
                                    5 <= len(title_candidate) <= 120 and
                                    any(kw in title_candidate.lower() for kw in product_keywords)):
                                    title = title_candidate
                                    break
                        if title:
                            break
                
                # Extract model - pattern like "Model MC700, MC724" or "Model MC700" or "Model A1234"
                model_pattern = r'Model\s+([A-Z0-9]+(?:\s*,\s*[A-Z0-9]+)*)'
                for line in lines[:100]:
                    match = re.search(model_pattern, line, re.IGNORECASE)
                    if match:
                        model = match.group(1).strip()
                        break
                
                # If no "Model" prefix found, try to find model codes directly (for iPhone, LED Cinema Display)
                if not model and title:
                    # For iPhone and LED Cinema Display, always try to find model
                    if "iphone" in title.lower() or "led cinema display" in title.lower() or "cinema display" in title.lower():
                        # Look for model patterns like A1234, MC700, etc.
                        model_pattern2 = r'\b([A-Z]{1,2}\d{3,4}(?:\s*,\s*[A-Z]{1,2}\d{3,4})*)\b'
                        for line in lines[:80]:
                            match = re.search(model_pattern2, line)
                            if match:
                                potential_model = match.group(1).strip()
                                # Make sure it's not part of a date or other number
                                if len(potential_model) >= 4:  # At least A123 format
                                    model = potential_model
                                    break
                
                # Return title and model (model can be None for some products)
                if title:
                    # Final cleanup of title
                    title = title.strip()
                    # Fix split numbers (e.g., "1 3-inch" -> "13-inch")
                    title = self.fix_split_numbers(title)
                    # Remove any trailing punctuation that doesn't belong
                    title = re.sub(r'[.,;:]+$', '', title)
                    return (title, model)
                
                return None
                
        except Exception as e:
            print(f"  [ERROR] Failed to read PDF {pdf_path.name}: {e}")
            return None

    def rename_pdfs(self) -> Dict[str, int]:
        """
        Rename PDFs based on their content (title and model).
        Renames all PDFs except those starting with "Read Our".
        
        Returns:
            Dictionary with counts of renamed, skipped, and error files
        """
        if not self.output_dir.exists():
            print(f"Output directory {self.output_dir} does not exist.")
            return {"renamed": 0, "skipped": 0, "errors": 0}
        
        pdf_files = list(self.output_dir.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found to rename.")
            return {"renamed": 0, "skipped": 0, "errors": 0}
        
        print(f"\n{'='*60}")
        print(f"Renaming PDFs based on content...")
        print(f"{'='*60}\n")
        
        stats = {"renamed": 0, "skipped": 0, "errors": 0}
        
        for pdf_path in pdf_files:
            try:
                # Skip files that start with "Read Our"
                if pdf_path.name.startswith("Read Our") or pdf_path.name.startswith("Read our"):
                    stats["skipped"] += 1
                    continue
                
                # Check if filename already has split numbers (e.g., "1 3-inch" -> "13-inch")
                # If so, fix it first before extracting
                current_name = pdf_path.stem  # filename without .pdf extension
                fixed_name = self.fix_split_numbers(current_name)
                if fixed_name != current_name:
                    # Filename has split numbers, fix it
                    new_fixed_filename = f"{fixed_name}.pdf"
                    new_fixed_path = self.output_dir / self.sanitize_filename(new_fixed_filename)
                    if new_fixed_path != pdf_path and not new_fixed_path.exists():
                        pdf_path.rename(new_fixed_path)
                        pdf_path = new_fixed_path
                        print(f"  [FIXED] Fixed split numbers in filename: {current_name}.pdf -> {new_fixed_filename}")
                
                result = self.extract_title_and_model_from_pdf(pdf_path)
                
                if result is None:
                    # Extraction failed or no title found - skip
                    print(f"  [SKIP] {pdf_path.name} (could not extract title)")
                    stats["skipped"] += 1
                    continue
                
                title, model = result
                
                # Create new filename based on whether model exists
                if model:
                    # For iPhone and LED Cinema Display, always include model
                    if "iphone" in title.lower() or "led cinema display" in title.lower() or "cinema display" in title.lower():
                        new_filename = f"{title} (Model {model}).pdf"
                    # For other products, include model if available
                    else:
                        new_filename = f"{title} (Model {model}).pdf"
                else:
                    # No model found, just use title
                    new_filename = f"{title}.pdf"
                
                new_filename = self.sanitize_filename(new_filename)
                new_path = self.output_dir / new_filename
                
                # Skip if already renamed
                if pdf_path.name == new_filename:
                    stats["skipped"] += 1
                    continue
                
                # Check if target filename already exists
                if new_path.exists() and new_path != pdf_path:
                    # If target exists, try to add a distinguishing suffix (like date from original filename)
                    # Extract date/month from original filename if possible
                    original_name = pdf_path.stem
                    date_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', original_name, re.IGNORECASE)
                    if date_match:
                        month_year = f"{date_match.group(1)} {date_match.group(2)}"
                        # Insert date before .pdf
                        new_filename_with_date = f"{title} ({month_year})"
                        if model:
                            new_filename_with_date = f"{title} (Model {model}, {month_year})"
                        new_filename_with_date = f"{new_filename_with_date}.pdf"
                        new_filename_with_date = self.sanitize_filename(new_filename_with_date)
                        new_path_with_date = self.output_dir / new_filename_with_date
                        
                        if not new_path_with_date.exists():
                            new_filename = new_filename_with_date
                            new_path = new_path_with_date
                        else:
                            print(f"  [SKIP] {pdf_path.name} -> (target exists: {new_filename})")
                            stats["skipped"] += 1
                            continue
                    else:
                        # Try to extract year from filename
                        year_match = re.search(r'\((\d{4})\s+PDF\)', original_name)
                        if year_match:
                            year = year_match.group(1)
                            new_filename_with_year = f"{title} ({year})"
                            if model:
                                new_filename_with_year = f"{title} (Model {model}, {year})"
                            new_filename_with_year = f"{new_filename_with_year}.pdf"
                            new_filename_with_year = self.sanitize_filename(new_filename_with_year)
                            new_path_with_year = self.output_dir / new_filename_with_year
                            
                            if not new_path_with_year.exists():
                                new_filename = new_filename_with_year
                                new_path = new_path_with_year
                            else:
                                print(f"  [SKIP] {pdf_path.name} -> (target exists: {new_filename})")
                                stats["skipped"] += 1
                                continue
                        else:
                            print(f"  [SKIP] {pdf_path.name} -> (target exists: {new_filename})")
                            stats["skipped"] += 1
                            continue
                
                # Rename the file
                pdf_path.rename(new_path)
                print(f"  [RENAMED] {pdf_path.name} -> {new_filename}")
                stats["renamed"] += 1
                
            except Exception as e:
                print(f"  [ERROR] Failed to rename {pdf_path.name}: {e}")
                stats["errors"] += 1
        
        return stats


def main():
    """Main execution function."""
    scraper = AppleScraper()
    
    # You can add your start URLs here or pass them as command line arguments
    if len(sys.argv) > 1:
        # URLs provided as command line arguments
        start_urls = sys.argv[1:]
    else:
        # Default URLs - edit these or pass URLs as command line arguments
        start_urls = [
            "https://www.apple.com/environment/reports/"
        ]
    
    if not start_urls:
        print("Error: No start URLs provided.")
        print("\nUsage:")
        print("  python apple.py <url1> <url2> ...")
        print("\nOr edit apple.py and add URLs to the start_urls list.")
        sys.exit(1)
    
    print(f"Starting Apple PDF scraper with {len(start_urls)} URL(s)...")
    print(f"Output directory: {scraper.output_dir}\n")
    
    try:
        metadata = scraper.crawl(start_urls)
        
        # Count successful downloads
        downloaded = sum(1 for m in metadata if m.get("status") == "downloaded")
        skipped = sum(1 for m in metadata if m.get("status") == "skipped")
        errors = sum(1 for m in metadata if m.get("status") == "error")
        
        print(f"\n{'='*60}")
        print(f"Scraping completed!")
        print(f"  Downloaded: {downloaded}")
        print(f"  Skipped (already exists): {skipped}")
        print(f"  Errors: {errors}")
        print(f"  Total processed: {len(metadata)}")
        print(f"{'='*60}")
        
        if errors > 0:
            print("\nErrors encountered:")
            for m in metadata:
                if m.get("status") == "error":
                    print(f"  - {m.get('url', 'unknown')}: {m.get('error', 'unknown error')}")
        
        # Rename PDFs based on their content
        rename_stats = scraper.rename_pdfs()
        if rename_stats["renamed"] > 0 or rename_stats["skipped"] > 0 or rename_stats["errors"] > 0:
            print(f"\n{'='*60}")
            print(f"Renaming completed!")
            print(f"  Renamed: {rename_stats['renamed']}")
            print(f"  Skipped: {rename_stats['skipped']}")
            print(f"  Errors: {rename_stats['errors']}")
            print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

