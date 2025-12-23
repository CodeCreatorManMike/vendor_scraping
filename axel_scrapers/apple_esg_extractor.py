
import argparse
import csv
import io
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pdfplumber
import pypdfium2 as pdfium
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

# ============================================================
# HARD-CODED API KEY (per your instruction)
# ============================================================
OPENAI_API_KEY = ""

# ============================================================
# MODEL CONFIG
# ============================================================
MODEL = "gpt-4o-mini"  # vision-capable and cost-effective

# ============================================================
# CSV SCHEMA (MUST MATCH Lenovo ESG CSV EXACTLY)
# ============================================================
CSV_COLUMNS = [
    "FileName",
    "Hardware_Model",
    "Energy_Star",
    "Estimated_Impact_Mean_kgCO2e",
    "Estimated_Impact_Low_kgCO2e",
    "Estimated_Impact_High_kgCO2e",
    "Manufacturing_or_Production_%",
    "Distribution_%",
    "Use_%",
    "End_of_Life_%",
    "Mainboard_%",
    "SSD_%",
    "Display_%",
    "Chassis_%",
    "Batteries_%",
    "PSU_%",
    "Transport_%",
    "Packaging_%",
    "Others_%",
    "Lifetime_years",
    "Use_location",
    "Use_energy_kWh_per_year",
    "source_pdf_path",
]

# ============================================================
# LOGGING SETUP (clear terminal reporting)
# ============================================================
def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("apple_esg_extractor")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    # Prevent duplicate handlers if re-run in same session
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


# ============================================================
# SMALL UTILITIES
# ============================================================
def normalize_csv_path(p: Path) -> str:
    """
    Lenovo CSV uses backslashes in source_pdf_path (Windows style).
    To keep your outputs consistent, we normalize to backslashes.
    """
    return str(p).replace("/", "\\")


def safe_float(x: Any) -> Optional[float]:
    """Convert common numeric representations into float, else None."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    # remove commas and common units
    s = s.replace(",", "")
    s = re.sub(r"(kg\s*co2e|kgco2e|co2e|%)", "", s, flags=re.I).strip()
    try:
        return float(s)
    except Exception:
        return None


def clamp_percent(x: Optional[float]) -> Optional[float]:
    """Keep percentages in a sane 0-100 range if parse is odd."""
    if x is None:
        return None
    if x < 0:
        return 0.0
    if x > 100:
        return 100.0
    return x


def extract_json_from_model_output(s: str) -> Any:
    """
    Robust JSON parser:
    - try direct json.loads
    - fallback: extract the first JSON object/array from the text
    """
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    # If the model accidentally returns extra text, grab JSON block.
    m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
    if not m:
        raise ValueError("No JSON object/array found in model output.")
    return json.loads(m.group(1))


def pil_to_data_url(img: Image.Image) -> str:
    """Convert PIL image -> base64 PNG data URL for OpenAI vision input."""
    import base64

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ============================================================
# EXISTING-ROWS SKIP LOGIC (only process PDFs you haven't touched)
# ============================================================
def load_processed_keys(output_csv: Path, logger: logging.Logger) -> Tuple[Set[str], Set[str]]:
    """
    We skip work based on what's already in apple_esg.csv.

    Because Apple PDFs can yield multiple rows (multiple products per file),
    we must treat a PDF as "done" if ANY row exists with same:
      - FileName, or
      - source_pdf_path

    Returns:
      processed_filenames, processed_source_paths
    """
    if not output_csv.exists():
        return set(), set()

    try:
        df = pd.read_csv(output_csv)
    except Exception as e:
        logger.warning(f"Could not read existing CSV for skip logic: {output_csv} -> {e}")
        return set(), set()

    processed_filenames = set()
    processed_source_paths = set()

    if "FileName" in df.columns:
        processed_filenames = set(df["FileName"].astype(str).fillna("").tolist())
    if "source_pdf_path" in df.columns:
        processed_source_paths = set(df["source_pdf_path"].astype(str).fillna("").tolist())

    # Normalize path slashes for robust comparison
    processed_source_paths = {p.replace("/", "\\") for p in processed_source_paths}

    return processed_filenames, processed_source_paths


# ============================================================
# PDF EXTRACTION (text + tables + candidate page detection)
# ============================================================
@dataclass
class ExtractedPdfContent:
    text: str
    candidate_pages: List[int]  # 0-based page indices that likely contain emissions charts


def extract_text_tables_and_candidates(
    pdf_path: Path,
    text_char_limit: int,
    logger: logging.Logger,
) -> ExtractedPdfContent:
    """
    Uses pdfplumber to extract:
      - text
      - tables (flattened into pipe-separated rows)
    Also identifies "candidate pages" that likely contain the key device footprint chart/table.

    Candidate detection is intentionally heuristic, because Apple PDFs vary:
      - search for "kg CO2e", "CO2e", "carbon footprint", "life cycle"
      - search for stage labels like "Production", "Transportation", "End-of-life", "Product use"
    """
    chunks: List[str] = []
    candidate_pages: List[int] = []

    # Patterns that often appear near the Apple footprint bar
    key_patterns = [
        r"\bkg\s*CO2e\b",
        r"\bCO2e\b",
        r"carbon\s+footprint",
        r"life\s+cycle",
        r"\bProduction\b",
        r"\bTransportation\b",
        r"End[-\s]?of[-\s]?life",
        r"Product\s+use",
    ]
    key_re = re.compile("|".join(key_patterns), re.I)

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                chunks.append(page_text)

            # tables -> pipe-separated rows
            try:
                tables = page.extract_tables() or []
                for t in tables:
                    for row in t:
                        if not row:
                            continue
                        row_clean = [str(c).strip() for c in row if c and str(c).strip()]
                        if row_clean:
                            chunks.append(" | ".join(row_clean))
            except Exception:
                # Table extraction can fail on some pages; ignore and continue.
                pass

            # Candidate page heuristic
            if key_re.search(page_text):
                candidate_pages.append(i)

    text = "\n".join(chunks)
    if len(text) > text_char_limit:
        text = text[:text_char_limit]

    # If we found no candidates, default to first few pages (still useful)
    if not candidate_pages:
        candidate_pages = list(range(0, 3))

    # Deduplicate while preserving order
    seen = set()
    candidate_pages = [p for p in candidate_pages if not (p in seen or seen.add(p))]

    logger.debug(f"Candidate pages for charts/tables: {candidate_pages}")
    return ExtractedPdfContent(text=text, candidate_pages=candidate_pages)


def render_pdf_pages_as_images(
    pdf_path: Path,
    page_indices: List[int],
    max_pages: int,
    logger: logging.Logger,
) -> List[Image.Image]:
    """
    Rasterize selected pages to images using pdfium.
    We render only a limited number of pages (max_pages) to keep requests fast & consistent.

    This is critical for Apple PDFs where the key emissions chart is sometimes embedded as graphics
    and not fully captured in text extraction.
    """
    images: List[Image.Image] = []

    try:
        doc = pdfium.PdfDocument(str(pdf_path))
    except Exception as e:
        logger.warning(f"Could not open PDF with pdfium: {pdf_path} -> {e}")
        return images

    # Bound to document length
    valid_indices = [i for i in page_indices if 0 <= i < len(doc)]
    if not valid_indices:
        valid_indices = list(range(0, min(len(doc), max_pages)))

    # Take up to max_pages pages
    valid_indices = valid_indices[:max_pages]

    for idx in valid_indices:
        try:
            page = doc[idx]
            bitmap = page.render(scale=2.5)
            img = bitmap.to_pil()
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        except Exception as e:
            logger.warning(f"Render failed for {pdf_path.name} page {idx + 1}: {e}")
            continue

    return images


# ============================================================
# AI EXTRACTION PROMPTING
# ============================================================
def call_openai_with_retries(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    logger: logging.Logger,
    max_tokens: int = 1600,
    temperature: float = 0.0,
    retries: int = 3,
    backoff_seconds: float = 2.0,
) -> str:
    """
    OpenAI call wrapper with retry/backoff.
    This makes large batch runs far more stable (temporary network hiccups etc).
    """
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            logger.warning(f"OpenAI call failed (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)

    raise RuntimeError(f"OpenAI call failed after {retries} attempts: {last_err}")


def extract_esg_records_with_ai(
    client: OpenAI,
    extracted_text: str,
    images: List[Image.Image],
    pdf_name: str,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """
    Returns a LIST of product records.
    - If the PDF contains a single device: list length = 1
    - If it contains multiple devices (common for Apple reports): list length > 1

    We normalize to "Lenovo-like" fields so you can keep your downstream flow consistent.
    """
    system_prompt = (
        "You are an ESG extraction system.\n"
        "You will receive text + images from an Apple environmental report PDF.\n"
        "These PDFs may describe ONE product or MULTIPLE products.\n\n"
        "Output ONLY valid JSON.\n"
        "If multiple products are present, output an array of objects (one per product).\n"
        "If only one product is present, you may output a single object.\n"
        "Do NOT output markdown and do NOT include commentary."
    )

    # We ask for a practical schema that maps cleanly into your Lenovo-like CSV columns.
    schema = {
        "products": [
            {
                "Hardware_Model": "string (product/device name; required)",
                "model_year": "string/int or null",
                "Estimated_Impact_Mean_kgCO2e": "number or null (total footprint)",
                "Estimated_Impact_Low_kgCO2e": "number or null",
                "Estimated_Impact_High_kgCO2e": "number or null",
                "Manufacturing_or_Production_%": "number or null",
                "Distribution_%": "number or null (if present; else null)",
                "Use_%": "number or null",
                "End_of_Life_%": "number or null",
                "Transport_%": "number or null (Apple 'Transportation' often maps here)",
                "Packaging_%": "number or null",
                "Others_%": "number or null",
                "Lifetime_years": "number or null",
                "Use_location": "string or null (if mentioned)",
                "Use_energy_kWh_per_year": "number or null",
                "Energy_Star": "string/bool or null",
                "notes": "string or null (brief: materials/recycled/packaging/other ESG facts)"
            }
        ],
        "rules": [
            "If the PDF shows a chart with stage % splits: map them to the % fields.",
            "Apple reports often have 'Production', 'Transportation', 'Product use', 'End-of-life'.",
            "Put Apple 'Transportation' into Transport_% (and Distribution_% can remain null unless explicitly present).",
            "If a product's total footprint is shown as 'XX kg CO2e', use that for Estimated_Impact_Mean_kgCO2e.",
            "If multiple products appear (e.g., multiple watch bands), return one object per product.",
        ],
    }

    user_parts: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"PDF filename: {pdf_name}\n\n"
                "Extract ESG/carbon footprint data and return JSON matching this schema:\n"
                + json.dumps(schema, indent=2)
                + "\n\n"
                "Here is extracted text/tables:\n"
                + (extracted_text or "")
            ),
        }
    ]

    # Add images for chart/table reading
    for img in images:
        user_parts.append({"type": "image_url", "image_url": {"url": pil_to_data_url(img)}})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_parts},
    ]

    raw = call_openai_with_retries(client, messages, logger=logger, max_tokens=1800)
    parsed = extract_json_from_model_output(raw)

    # Allow model to return either object or list; normalize to list of product dicts
    if isinstance(parsed, dict) and "products" in parsed:
        products = parsed.get("products") or []
    elif isinstance(parsed, list):
        products = parsed
    elif isinstance(parsed, dict):
        products = [parsed]
    else:
        raise ValueError("Model returned JSON that is not an object or array.")

    # Basic cleanup / normalization
    cleaned: List[Dict[str, Any]] = []
    for p in products:
        if not isinstance(p, dict):
            continue

        hw = (p.get("Hardware_Model") or "").strip()
        if not hw:
            # If model forgot to set Hardware_Model, skip it (we cannot write a useful row).
            continue

        cleaned.append(p)

    return cleaned


# ============================================================
# ROW BUILDING (map AI outputs into Lenovo-like columns)
# ============================================================
def build_csv_row(
    pdf_path: Path,
    product: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Map one AI product record -> one CSV row with EXACT Lenovo headings.
    Anything Apple does not provide is left blank/None.
    """
    # Apple "Transportation" may map to Transport_% (and Distribution_% usually blank)
    transport_pct = clamp_percent(safe_float(product.get("Transport_%")))
    dist_pct = clamp_percent(safe_float(product.get("Distribution_%")))

    row = {c: None for c in CSV_COLUMNS}

    row["FileName"] = pdf_path.name
    row["Hardware_Model"] = product.get("Hardware_Model")
    row["Energy_Star"] = product.get("Energy_Star")

    row["Estimated_Impact_Mean_kgCO2e"] = safe_float(product.get("Estimated_Impact_Mean_kgCO2e"))
    row["Estimated_Impact_Low_kgCO2e"] = safe_float(product.get("Estimated_Impact_Low_kgCO2e"))
    row["Estimated_Impact_High_kgCO2e"] = safe_float(product.get("Estimated_Impact_High_kgCO2e"))

    row["Manufacturing_or_Production_%"] = clamp_percent(
        safe_float(product.get("Manufacturing_or_Production_%"))
    )
    row["Distribution_%"] = dist_pct
    row["Use_%"] = clamp_percent(safe_float(product.get("Use_%")))
    row["End_of_Life_%"] = clamp_percent(safe_float(product.get("End_of_Life_%")))

    # These component breakdown fields are Lenovo-specific; Apple often won't have them.
    # We keep them blank unless the model explicitly returns them.
    row["Mainboard_%"] = clamp_percent(safe_float(product.get("Mainboard_%")))
    row["SSD_%"] = clamp_percent(safe_float(product.get("SSD_%")))
    row["Display_%"] = clamp_percent(safe_float(product.get("Display_%")))
    row["Chassis_%"] = clamp_percent(safe_float(product.get("Chassis_%")))
    row["Batteries_%"] = clamp_percent(safe_float(product.get("Batteries_%")))
    row["PSU_%"] = clamp_percent(safe_float(product.get("PSU_%")))

    row["Transport_%"] = transport_pct
    row["Packaging_%"] = clamp_percent(safe_float(product.get("Packaging_%")))
    row["Others_%"] = clamp_percent(safe_float(product.get("Others_%")))

    row["Lifetime_years"] = safe_float(product.get("Lifetime_years"))
    row["Use_location"] = product.get("Use_location")
    row["Use_energy_kWh_per_year"] = safe_float(product.get("Use_energy_kWh_per_year"))

    # Store full path (normalized like Lenovo)
    row["source_pdf_path"] = normalize_csv_path(pdf_path)

    return row


# ============================================================
# MAIN WORKFLOW
# ============================================================
def find_apple_pdfs(base_data_dir: Path) -> List[Path]:
    """
    Finds all PDFs under:
      data/pdfs/apple/**/*.pdf
    """
    apple_dir = base_data_dir / "apple"
    if not apple_dir.exists():
        return []
    return list(apple_dir.rglob("*.pdf"))


def append_rows_to_csv(output_csv: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Append rows to output CSV, writing header if file doesn't exist.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists()

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
        f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apple ESG PDF extractor (Lenovo-style CSV).")

    parser.add_argument("--base-data-dir", default="data/pdfs", help="Base folder containing vendor pdf folders.")
    parser.add_argument("--output", default="data/metadata/apple_esg.csv", help="Output CSV path.")
    parser.add_argument("--single-pdf", default=None, help="Process just one PDF file path.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip PDFs already present in the output CSV.")
    parser.add_argument("--max-image-pages", type=int, default=6, help="Max pages to render as images per PDF.")
    parser.add_argument("--text-char-limit", type=int, default=24000, help="Max characters of extracted text/tables.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")

    args = parser.parse_args()
    logger = setup_logger(args.verbose)

    base_dir = Path(args.base_data_dir)
    output_csv = Path(args.output)

    # OpenAI client (key is hardcoded above)
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Load "already processed" sets so we can skip work reliably
    processed_filenames, processed_source_paths = load_processed_keys(output_csv, logger)

    # Collect PDFs to process
    if args.single_pdf:
        pdfs = [Path(args.single_pdf)]
    else:
        pdfs = find_apple_pdfs(base_dir)

    if not pdfs:
        logger.error(f"No PDFs found. Expected under: {base_dir / 'apple'}")
        sys.exit(1)

    # Summary counters for end-of-run reporting
    total = len(pdfs)
    skipped = 0
    success_files = 0
    success_rows = 0
    failed = 0

    logger.info(f"Found {total} Apple PDF(s). Output -> {output_csv}")
    if args.skip_existing and output_csv.exists():
        logger.info(
            f"Skip-existing enabled. Already have {len(processed_filenames)} FileName(s) "
            f"and {len(processed_source_paths)} source_pdf_path(s) recorded."
        )

    # Process each PDF
    for pdf_path in tqdm(pdfs, desc="Extracting Apple ESG", unit="pdf"):
        try:
            if not pdf_path.exists():
                logger.warning(f"Missing file (skipping): {pdf_path}")
                skipped += 1
                continue

            pdf_key_name = pdf_path.name
            pdf_key_path = normalize_csv_path(pdf_path)

            # Skip if we've already touched this PDF (by name OR source path)
            if args.skip_existing and (pdf_key_name in processed_filenames or pdf_key_path in processed_source_paths):
                logger.debug(f"Skipping already-processed PDF: {pdf_key_name}")
                skipped += 1
                continue

            logger.info(f"Processing: {pdf_key_name}")

            # 1) Extract text + tables, and detect candidate pages likely containing the key chart/table
            extracted = extract_text_tables_and_candidates(
                pdf_path=pdf_path,
                text_char_limit=args.text_char_limit,
                logger=logger,
            )

            # 2) Render candidate pages to images for vision reading (charts/tables as graphics)
            images = render_pdf_pages_as_images(
                pdf_path=pdf_path,
                page_indices=extracted.candidate_pages,
                max_pages=args.max_image_pages,
                logger=logger,
            )

            # 3) Use AI to convert messy raw content -> normalized product records
            products = extract_esg_records_with_ai(
                client=client,
                extracted_text=extracted.text,
                images=images,
                pdf_name=pdf_path.name,
                logger=logger,
            )

            if not products:
                # We treat "no products found" as a failure to keep you aware.
                raise RuntimeError("AI returned no product records (empty result).")

            # 4) Convert each product -> a Lenovo-style CSV row
            new_rows: List[Dict[str, Any]] = [build_csv_row(pdf_path, p) for p in products]

            # 5) Append to CSV immediately (so you don't lose progress if a later PDF fails)
            append_rows_to_csv(output_csv, new_rows)

            # 6) Mark this PDF as processed so we don't re-run it in this session
            processed_filenames.add(pdf_key_name)
            processed_source_paths.add(pdf_key_path)

            success_files += 1
            success_rows += len(new_rows)
            logger.info(f"✅ Success: {pdf_key_name} -> wrote {len(new_rows)} row(s)")

        except Exception as e:
            failed += 1
            logger.error(f"❌ Failed: {pdf_path.name} -> {e}")

    # Final summary
    logger.info("============================================================")
    logger.info("Run summary")
    logger.info(f"Total PDFs discovered: {total}")
    logger.info(f"Skipped (already processed/missing): {skipped}")
    logger.info(f"Successful PDFs: {success_files}")
    logger.info(f"Rows written: {success_rows}")
    logger.info(f"Failed PDFs: {failed}")
    logger.info(f"Output CSV: {output_csv}")
    logger.info("============================================================")


if __name__ == "__main__":
    main()
