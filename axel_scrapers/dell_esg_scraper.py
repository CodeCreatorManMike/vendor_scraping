
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
MODEL = "gpt-4o-mini"  # vision-capable

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
# LOGGING SETUP
# ============================================================
def setup_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("dell_esg_extractor")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


# ============================================================
# UTILITIES
# ============================================================
def normalize_csv_path(p: Path) -> str:
    """
    Lenovo-style paths are usually Windows-ish.
    Normalize slashes so your skip logic is stable on Windows.
    """
    return str(p).replace("/", "\\")


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    s = s.replace(",", "")
    s = re.sub(r"(kg\s*co2e|kgco2e|kg\s*co2\s*-?eq|co2\s*-?eq|co2e|%)", "", s, flags=re.I).strip()
    try:
        return float(s)
    except Exception:
        return None


def clamp_percent(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if x < 0:
        return 0.0
    if x > 100:
        return 100.0
    return x


def extract_json_from_model_output(s: str) -> Any:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
    if not m:
        raise ValueError("No JSON object/array found in model output.")
    return json.loads(m.group(1))


def pil_to_data_url(img: Image.Image) -> str:
    import base64

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ============================================================
# SKIP LOGIC (only process PDFs not already done)
# ============================================================
def load_processed_keys(output_csv: Path, logger: logging.Logger) -> Tuple[Set[str], Set[str]]:
    """
    For Dell, most PDFs are one device per PDF, but some can be multi-page or multi-scenario.
    We treat the PDF as done if we already have ANY row with same FileName or source_pdf_path.
    """
    if not output_csv.exists():
        return set(), set()

    try:
        df = pd.read_csv(output_csv)
    except Exception as e:
        logger.warning(f"Could not read existing CSV for skip logic: {output_csv} -> {e}")
        return set(), set()

    processed_filenames = set(df["FileName"].astype(str).fillna("").tolist()) if "FileName" in df.columns else set()
    processed_source_paths = set(df["source_pdf_path"].astype(str).fillna("").tolist()) if "source_pdf_path" in df.columns else set()

    processed_source_paths = {p.replace("/", "\\") for p in processed_source_paths}
    return processed_filenames, processed_source_paths


# ============================================================
# PDF EXTRACTION (text + tables + candidate pages)
# ============================================================
@dataclass
class ExtractedPdfContent:
    text: str
    candidate_pages: List[int]  # 0-based indices


def extract_text_tables_and_candidates(
    pdf_path: Path,
    text_char_limit: int,
    logger: logging.Logger,
) -> ExtractedPdfContent:
    """
    Dell PDFs often contain:
    - "This product’s estimated carbon footprint: XXX kgCO2e +/- YYY kgCO2e"
    - pie chart with lifecycle stage percentages
    - breakout "Manufacturing by component" (Mainboard, SSD, Display, Chassis, Battery, PSU, Packaging, etc.)
    - assumptions table with lifetime, use location, energy demand (kWh/yr), etc.
    These are often captured by text extraction, but charts/tables can be images → we also render pages.
    """

    chunks: List[str] = []
    candidate_pages: List[int] = []

    # We bias candidates towards pages likely containing the chart + assumptions
    key_patterns = [
        r"estimated\s+carbon\s+footprint",
        r"total\s+carbon\s+footprint",
        r"\bkg\s*CO2e\b",
        r"CO2\s*-?eq",
        r"manufacturing\s+breakdown",
        r"Estimated\s+impact\s+by\s+lifecycle",
        r"\bUse\s+Location\b",
        r"Energy\s+Demand",
        r"Product\s+Lifetime",
        r"\bTransportation\b",
        r"\bEnd[-\s]?of[-\s]?life\b",
        r"\bPackaging\b",
        r"\bMainboard\b",
        r"\bSolid\s*State\s*Drive",
        r"\bChassis\b",
        r"\bDisplay\b",
        r"\bBattery\b",
        r"\bPower\s*Supply",
    ]
    key_re = re.compile("|".join(key_patterns), re.I)

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                chunks.append(page_text)

            # Table extraction
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
                pass

            if key_re.search(page_text):
                candidate_pages.append(i)

    text = "\n".join(chunks)
    if len(text) > text_char_limit:
        text = text[:text_char_limit]

    # If nothing hits, still render first pages
    if not candidate_pages:
        candidate_pages = list(range(0, 3))

    # Deduplicate preserve order
    seen = set()
    candidate_pages = [p for p in candidate_pages if not (p in seen or seen.add(p))]

    logger.debug(f"Candidate pages: {candidate_pages}")
    return ExtractedPdfContent(text=text, candidate_pages=candidate_pages)


def render_pdf_pages_as_images(
    pdf_path: Path,
    page_indices: List[int],
    max_pages: int,
    logger: logging.Logger,
) -> List[Image.Image]:
    """
    Render selected pages using pdfium for vision-based extraction.
    This is what makes the script consistent on charts and graphical tables.
    """
    images: List[Image.Image] = []

    try:
        doc = pdfium.PdfDocument(str(pdf_path))
    except Exception as e:
        logger.warning(f"Could not open PDF with pdfium: {pdf_path} -> {e}")
        return images

    valid_indices = [i for i in page_indices if 0 <= i < len(doc)]
    if not valid_indices:
        valid_indices = list(range(0, min(len(doc), max_pages)))

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

    return images


# ============================================================
# OPENAI CALLS (with retry)
# ============================================================
def call_openai_with_retries(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    logger: logging.Logger,
    max_tokens: int = 1700,
    temperature: float = 0.0,
    retries: int = 3,
    backoff_seconds: float = 2.0,
) -> str:
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


# ============================================================
# AI EXTRACTION (Dell-specific hints, Lenovo-style output)
# ============================================================
def extract_esg_records_with_ai(
    client: OpenAI,
    extracted_text: str,
    images: List[Image.Image],
    pdf_name: str,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """
    Dell PDFs are usually one product per PDF, but can include:
    - +/- uncertainty (mean +/- range)
    - lifecycle stage percentages (Manufacturing, Transportation/Distribution, Use, End-of-life)
    - manufacturing component breakdown (Mainboard, SSD, Display, Chassis, Battery, PSU, Packaging, etc.)
    - assumptions: lifetime years, use location, energy demand kWh/yr, sometimes Energy Star mention

    We ask the model for an array of products anyway, to remain robust.
    """

    system_prompt = (
        "You are an ESG / Product Carbon Footprint extraction system.\n"
        "You will receive raw text + images from Dell PCF/LCA PDFs.\n"
        "Output ONLY valid JSON.\n"
        "If multiple products appear, return an array. If one product, return a single object.\n"
        "No markdown, no commentary."
    )

    # We want direct mapping into your Lenovo-style headings.
    # We also include Dell-specific guidance: Transportation vs Distribution naming differences.
    schema = {
        "products": [
            {
                "Hardware_Model": "string (required; e.g., 'Dell Inspiron 13 5320')",
                "Energy_Star": "string/bool/null (if referenced)",
                "Estimated_Impact_Mean_kgCO2e": "number or null",
                "Estimated_Impact_Low_kgCO2e": "number or null",
                "Estimated_Impact_High_kgCO2e": "number or null",

                "Manufacturing_or_Production_%": "number or null",
                "Distribution_%": "number or null",
                "Use_%": "number or null",
                "End_of_Life_%": "number or null",

                "Mainboard_%": "number or null",
                "SSD_%": "number or null",
                "Display_%": "number or null",
                "Chassis_%": "number or null",
                "Batteries_%": "number or null",
                "PSU_%": "number or null",
                "Transport_%": "number or null",
                "Packaging_%": "number or null",
                "Others_%": "number or null",

                "Lifetime_years": "number or null",
                "Use_location": "string or null",
                "Use_energy_kWh_per_year": "number or null"
            }
        ],
        "rules": [
            "Prefer values explicitly shown in charts/tables, not inferred.",
            "For total footprint expressed as 'X kgCO2e +/- Y', set Mean=X, Low=X-Y, High=X+Y (if numeric).",
            "Lifecycle stage percentages: map Manufacturing -> Manufacturing_or_Production_%.",
            "Dell sometimes uses Transportation% (map that to Transport_%). If Distribution% is shown, map to Distribution_%.",
            "Component breakdown shown as % of manufacturing: map Mainboard/SSD/Display/Chassis/Battery/PSU/Packaging accordingly.",
            "If a component doesn't exist in the Dell breakdown, leave it null.",
            "Assumptions table may include Product Lifetime (years), Use Location, and Energy Demand (Yearly TEC) kWh/yr."
        ]
    }

    user_parts: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"PDF filename: {pdf_name}\n\n"
                "Extract ESG/PCF/LCA data from this Dell PDF and return JSON matching this schema:\n"
                + json.dumps(schema, indent=2)
                + "\n\nExtracted text/tables:\n"
                + (extracted_text or "")
            ),
        }
    ]

    for img in images:
        user_parts.append({"type": "image_url", "image_url": {"url": pil_to_data_url(img)}})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_parts},
    ]

    raw = call_openai_with_retries(client, messages, logger=logger, max_tokens=1900)
    parsed = extract_json_from_model_output(raw)

    # Normalize into list of dict products
    if isinstance(parsed, dict) and "products" in parsed:
        products = parsed.get("products") or []
    elif isinstance(parsed, list):
        products = parsed
    elif isinstance(parsed, dict):
        products = [parsed]
    else:
        raise ValueError("Model returned JSON that is not an object or array.")

    cleaned: List[Dict[str, Any]] = []
    for p in products:
        if not isinstance(p, dict):
            continue
        hw = (p.get("Hardware_Model") or "").strip()
        if not hw:
            continue
        cleaned.append(p)

    if not cleaned:
        raise RuntimeError("AI returned no usable product records (missing Hardware_Model).")

    return cleaned


# ============================================================
# ROW BUILDING (map AI outputs -> Lenovo headings)
# ============================================================
def build_csv_row(pdf_path: Path, product: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map AI record into EXACT Lenovo-style headings.
    """
    row = {c: None for c in CSV_COLUMNS}

    row["FileName"] = pdf_path.name
    row["Hardware_Model"] = product.get("Hardware_Model")
    row["Energy_Star"] = product.get("Energy_Star")

    mean = safe_float(product.get("Estimated_Impact_Mean_kgCO2e"))
    low = safe_float(product.get("Estimated_Impact_Low_kgCO2e"))
    high = safe_float(product.get("Estimated_Impact_High_kgCO2e"))

    row["Estimated_Impact_Mean_kgCO2e"] = mean
    row["Estimated_Impact_Low_kgCO2e"] = low
    row["Estimated_Impact_High_kgCO2e"] = high

    row["Manufacturing_or_Production_%"] = clamp_percent(safe_float(product.get("Manufacturing_or_Production_%")))
    row["Distribution_%"] = clamp_percent(safe_float(product.get("Distribution_%")))
    row["Use_%"] = clamp_percent(safe_float(product.get("Use_%")))
    row["End_of_Life_%"] = clamp_percent(safe_float(product.get("End_of_Life_%")))

    row["Mainboard_%"] = clamp_percent(safe_float(product.get("Mainboard_%")))
    row["SSD_%"] = clamp_percent(safe_float(product.get("SSD_%")))
    row["Display_%"] = clamp_percent(safe_float(product.get("Display_%")))
    row["Chassis_%"] = clamp_percent(safe_float(product.get("Chassis_%")))
    row["Batteries_%"] = clamp_percent(safe_float(product.get("Batteries_%")))
    row["PSU_%"] = clamp_percent(safe_float(product.get("PSU_%")))

    row["Transport_%"] = clamp_percent(safe_float(product.get("Transport_%")))
    row["Packaging_%"] = clamp_percent(safe_float(product.get("Packaging_%")))
    row["Others_%"] = clamp_percent(safe_float(product.get("Others_%")))

    row["Lifetime_years"] = safe_float(product.get("Lifetime_years"))
    row["Use_location"] = product.get("Use_location")
    row["Use_energy_kWh_per_year"] = safe_float(product.get("Use_energy_kWh_per_year"))

    row["source_pdf_path"] = normalize_csv_path(pdf_path)

    return row


# ============================================================
# IO HELPERS
# ============================================================
def find_dell_pdfs(base_data_dir: Path) -> List[Path]:
    """
    Find all Dell PDFs under: data/pdfs/dell/**/*.pdf
    """
    dell_dir = base_data_dir / "dell"
    if not dell_dir.exists():
        return []
    return list(dell_dir.rglob("*.pdf"))


def append_rows_to_csv(output_csv: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Append rows to CSV (write header if new file).
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


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Dell ESG/PCF PDF extractor (Lenovo-style CSV).")
    parser.add_argument("--base-data-dir", default="data/pdfs", help="Base folder containing vendor pdf folders.")
    parser.add_argument("--output", default="data/metadata/dell_esg.csv", help="Output CSV path.")
    parser.add_argument("--single-pdf", default=None, help="Process one PDF path only.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip PDFs already recorded in output CSV.")
    parser.add_argument("--max-image-pages", type=int, default=6, help="Max pages to render as images per PDF.")
    parser.add_argument("--text-char-limit", type=int, default=26000, help="Max extracted text/table characters.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    logger = setup_logger(args.verbose)

    base_dir = Path(args.base_data_dir)
    output_csv = Path(args.output)

    # OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    processed_filenames, processed_source_paths = load_processed_keys(output_csv, logger)

    # Choose PDFs
    if args.single_pdf:
        pdfs = [Path(args.single_pdf)]
    else:
        pdfs = find_dell_pdfs(base_dir)

    if not pdfs:
        logger.error(f"No Dell PDFs found. Expected under: {base_dir / 'dell'}")
        sys.exit(1)

    total = len(pdfs)
    skipped = 0
    success_files = 0
    success_rows = 0
    failed = 0

    logger.info(f"Found {total} Dell PDF(s). Output -> {output_csv}")
    if args.skip_existing and output_csv.exists():
        logger.info(
            f"Skip-existing enabled. Already have {len(processed_filenames)} FileName(s) "
            f"and {len(processed_source_paths)} source_pdf_path(s)."
        )

    for pdf_path in tqdm(pdfs, desc="Extracting Dell ESG", unit="pdf"):
        try:
            if not pdf_path.exists():
                logger.warning(f"Missing file (skipping): {pdf_path}")
                skipped += 1
                continue

            pdf_key_name = pdf_path.name
            pdf_key_path = normalize_csv_path(pdf_path)

            if args.skip_existing and (pdf_key_name in processed_filenames or pdf_key_path in processed_source_paths):
                skipped += 1
                continue

            logger.info(f"Processing: {pdf_key_name}")

            # 1) Extract text/tables + candidate pages
            extracted = extract_text_tables_and_candidates(
                pdf_path=pdf_path,
                text_char_limit=args.text_char_limit,
                logger=logger,
            )

            # 2) Render candidate pages (vision)
            images = render_pdf_pages_as_images(
                pdf_path=pdf_path,
                page_indices=extracted.candidate_pages,
                max_pages=args.max_image_pages,
                logger=logger,
            )

            # 3) AI normalize into schema
            products = extract_esg_records_with_ai(
                client=client,
                extracted_text=extracted.text,
                images=images,
                pdf_name=pdf_path.name,
                logger=logger,
            )

            # 4) Build rows
            rows = [build_csv_row(pdf_path, p) for p in products]

            # 5) Append immediately (crash-safe)
            append_rows_to_csv(output_csv, rows)

            # Mark processed
            processed_filenames.add(pdf_key_name)
            processed_source_paths.add(pdf_key_path)

            success_files += 1
            success_rows += len(rows)
            logger.info(f"✅ Success: {pdf_key_name} -> wrote {len(rows)} row(s)")

        except Exception as e:
            failed += 1
            logger.error(f"❌ Failed: {pdf_path.name} -> {e}")

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
