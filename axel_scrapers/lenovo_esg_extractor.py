import os
import io
import csv
import json
import base64
import argparse
from typing import List, Dict, Any

import pdfplumber
import pypdfium2 as pdfium
from openai import OpenAI


# ----------------------------
# HARD-CODED API KEY
# ----------------------------

OPENAI_API_KEY = ""  # PUT THE GPT API KEY HERE


# ----------------------------
# Configuration
# ----------------------------

VENDOR_NAME = "lenovo"

# PDFs live in data/pdfs/lenovo relative to this script
DEFAULT_BASE_DATA_DIR = os.path.join("data", "pdfs")

# CSV will be written to data/metadata/lenovo_esg.csv
DEFAULT_OUTPUT_CSV = os.path.join("data", "metadata", f"{VENDOR_NAME}_esg.csv")

# Limit how much text we send per PDF to keep token usage sane
MAX_TEXT_CHARS = 15000

# Model: you asked for "gpt 5" – this is the variable you control.
# If your account does NOT have a model literally called "gpt-5.1",
# change this to a real multimodal model you do have, e.g. "gpt-4.1".
OPENAI_MODEL = "gpt-5.1"


# ----------------------------
# Helpers: filesystem
# ----------------------------

def find_vendor_pdf_paths(
    base_dir: str = DEFAULT_BASE_DATA_DIR,
    vendor: str = VENDOR_NAME,
) -> List[str]:
    """
    Find all PDF files under <base_dir>/<vendor>/ (recursively).
    Example with your setup:
      base_dir = "data/pdfs"
      vendor   = "lenovo"
      -> data/pdfs/lenovo/*.pdf
    """
    vendor_dir = os.path.join(base_dir, vendor)
    pdf_paths: List[str] = []

    if not os.path.isdir(vendor_dir):
        raise FileNotFoundError(f"Expected directory {vendor_dir} with downloaded PDFs.")

    for root, _dirs, files in os.walk(vendor_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, fn))

    pdf_paths.sort()
    return pdf_paths


def ensure_output_dir(output_csv_path: str) -> None:
    """
    Make sure the parent directory of the CSV exists.
    """
    out_dir = os.path.dirname(output_csv_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def load_processed_pdf_paths(output_csv_path: str) -> set[str]:
    """
    If the output CSV already exists, load all existing source_pdf_path values
    so we can skip PDFs we've already processed.
    """
    processed: set[str] = set()
    if not os.path.isfile(output_csv_path):
        return processed

    try:
        with open(output_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row.get("source_pdf_path")
                if path:
                    processed.add(path)
    except Exception as e:
        print(f"[WARN] Could not read existing CSV {output_csv_path}: {e}")

    return processed


# ----------------------------
# Helpers: PDF parsing (TEXT)
# ----------------------------

def extract_text_and_tables(pdf_path: str) -> str:
    """
    Use pdfplumber to pull out raw text + table contents as a big string.
    This gives the model as much textual signal as possible.
    """
    parts: List[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                parts.append(f"--- PAGE {page_idx + 1} TEXT ---")
                parts.append(text)

            # Extract tables and render them as TSV-like text
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []

            for t_idx, table in enumerate(tables):
                parts.append(f"--- PAGE {page_idx + 1} TABLE {t_idx + 1} ---")
                for row in table:
                    if not row:
                        continue
                    row_cells = [
                        c.strip().replace("\n", " ") if isinstance(c, str) else ""
                        for c in row
                    ]
                    parts.append("\t".join(row_cells))

    combined = "\n".join(parts)
    if len(combined) > MAX_TEXT_CHARS:
        combined = combined[:MAX_TEXT_CHARS] + "\n...[TRUNCATED]..."
    return combined


# ----------------------------
# Helpers: PDF parsing (IMAGES / VISUALS)
# ----------------------------

def extract_page_images_as_base64(
    pdf_path: str,
    max_pages: int = 2,
) -> List[str]:
    """
    Use pypdfium2 to render the first few pages as PNG images,
    and return them as base64-encoded strings.

    We don't try to crop to just the pie charts; instead we render
    the whole page so the model can see *all* charts / diagrams.
    """
    b64_images: List[str] = []

    try:
        doc = pdfium.PdfDocument(pdf_path)
    except Exception as e:
        print(f"[WARN] Could not open PDF with pypdfium2: {pdf_path} ({e})")
        return b64_images

    try:
        page_count = len(doc)
        for page_index in range(min(page_count, max_pages)):
            try:
                page = doc.get_page(page_index)
                # Render at 2x scale for better readability of small text / charts
                bitmap = page.render(scale=2.0)
                pil_image = bitmap.to_pil()

                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                b64_images.append(b64)

                # Clean up
                bitmap.close()
                page.close()
            except Exception as e:
                print(f"[WARN] Failed to render page {page_index} of {pdf_path}: {e}")
                continue
    finally:
        doc.close()

    return b64_images


# ----------------------------
# Helpers: OpenAI call
# ----------------------------

def build_openai_client() -> OpenAI:
    """
    Build an OpenAI client using the hard-coded API key.
    """
    key = OPENAI_API_KEY
    if not key or key == "YOUR_OPENAI_API_KEY_HERE":
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put your real key in OPENAI_API_KEY at the top of this file."
        )
    return OpenAI(api_key=key)


def call_openai_for_esg_record(
    client: OpenAI,
    pdf_text: str,
    images_b64: List[str],
    pdf_path: str,
) -> Dict[str, Any]:
    """
    Ask the OpenAI model to read the Lenovo PCF sheet TEXT + IMAGES
    and return a *single* JSON object with a fixed schema suitable for a CSV row,
    aligned with the Dell/HP-style example you provided.
    """
    instructions = f"""
You are an ESG data extraction assistant.

You will be given the TEXT and PAGE IMAGES of a Lenovo Product Carbon Footprint (PCF) information sheet PDF.
Extract a single structured ESG record summarising the product.

You MUST use BOTH:
- The raw text / tables
- The visual content in charts / graphs / pie charts / diagrams

Focus on numeric and categorical values that are directly visible in the document
(including text, tables, legends, labels around pie charts, etc.).

Map the Lenovo PCF data into this schema (all keys MUST be present):

{{
  "FileName": string,                     # e.g. "ThinkPad-X1-Carbon.pdf"
  "Hardware_Model": string or null,       # commercial name / model name
  "Energy_Star": string or null,          # "Yes", "No", or null if unknown

  # Carbon footprint values (kg CO2e)
  "Estimated_Impact_Mean_kgCO2e": float or null,
  "Estimated_Impact_Low_kgCO2e": float or null,
  "Estimated_Impact_High_kgCO2e": float or null,

  # Life cycle stage contributions (percent of total impact)
  "Manufacturing_or_Production_%": float or null,
  "Distribution_%": float or null,        # also called transport stage, logistics, etc.
  "Use_%": float or null,
  "End_of_Life_%": float or null,

  # Component contributions (percent of total impact, if available)
  "Mainboard_%": float or null,
  "SSD_%": float or null,
  "Display_%": float or null,
  "Chassis_%": float or null,
  "Batteries_%": float or null,
  "PSU_%": float or null,
  "Transport_%": float or null,          # if transport is listed as a component slice
  "Packaging_%": float or null,
  "Others_%": float or null,

  # Usage assumptions
  "Lifetime_years": float or null,
  "Use_location": string or null,
  "Use_energy_kWh_per_year": float or null
}}

Interpretation rules for Lenovo PCF documents:
- If the sheet gives 5th, mean, and 95th percentile values:
  - Put mean in "Estimated_Impact_Mean_kgCO2e"
  - Put 5th percentile in "Estimated_Impact_Low_kgCO2e"
  - Put 95th percentile in "Estimated_Impact_High_kgCO2e"
- If there is only a single PCF value, treat it as "Estimated_Impact_Mean_kgCO2e"
  and set Low/High to null unless explicit bounds are given.
- For life cycle breakdowns, look for percentages associated with manufacturing/production,
  transport/distribution, use, and end-of-life and map them to the corresponding fields.
  Use BOTH text and pie chart visuals to confirm the numbers.
- For component breakdowns (mainboard, SSD, display, etc.), map the percentages listed
  for those components to the corresponding fields. Again, use BOTH tables and pie charts.
- Percent values should be numeric with no % sign, e.g. 37.5 for 37.5%.
- All numeric values should use a dot as decimal separator.
- DO NOT invent values. If you are not reasonably sure or the value is not present, use null.
- Always produce valid JSON, with double quotes around all keys and string values, and no trailing commas.
"""

    # Build multimodal content: text instructions + PDF text + images
    content: List[Dict[str, Any]] = []

    content.append(
        {
            "type": "text",
            "text": instructions,
        }
    )

    content.append(
        {
            "type": "text",
            "text": f"PDF FILE NAME: {os.path.basename(pdf_path)}\n\nPDF TEXT:\n{pdf_text}",
        }
    )

    for b64 in images_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}"
                },
            }
        )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a precise ESG data extraction engine that only outputs JSON.",
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[WARN] Could not parse JSON response for {pdf_path}. Raw content:")
        print(raw)
        raise

    # Ensure FileName is always set, even if the model forgot
    data.setdefault("FileName", os.path.basename(pdf_path))

    return data


# ----------------------------
# CSV handling
# ----------------------------

CSV_FIELDNAMES: List[str] = [
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


def normalise_row(raw: Dict[str, Any], pdf_path: str) -> Dict[str, Any]:
    """
    Flatten/normalise the JSON data from the model into a CSV row
    with the fixed columns above.
    """
    row: Dict[str, Any] = {}

    for field in CSV_FIELDNAMES:
        row[field] = None

    # FileName and source path
    row["FileName"] = raw.get("FileName") or os.path.basename(pdf_path)
    row["source_pdf_path"] = pdf_path

    # Direct scalar fields
    scalar_fields = [
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
    ]

    for f in scalar_fields:
        if f in raw:
            row[f] = raw[f]

    return row


# ----------------------------
# Main CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ESG numeric information from Lenovo PCF PDFs under data/pdfs/lenovo using the OpenAI API (text + visuals)."
    )
    parser.add_argument(
        "--base-data-dir",
        default=DEFAULT_BASE_DATA_DIR,
        help="Base data directory (default: ./data/pdfs). Lenovo PDFs are expected under data/pdfs/lenovo.",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help="Path to output CSV (default: data/metadata/lenovo_esg.csv).",
    )
    parser.add_argument(
        "--single-pdf",
        help="Path to a single PDF file to process instead of scanning the Lenovo directory.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If set (and not in single-pdf mode), skip PDFs whose source_pdf_path already exists in the output CSV.",
    )

    args = parser.parse_args()

    print(f"[INFO] Using base data dir: {args.base_data_dir}")
    print(f"[INFO] Output CSV will be: {args.output_csv}")

    # Decide which PDFs to process
    if args.single_pdf:
        pdf_paths = [args.single_pdf]
        print(f"[INFO] Running in single-PDF mode on: {args.single_pdf}")
    else:
        pdf_paths = find_vendor_pdf_paths(base_dir=args.base_data_dir, vendor=VENDOR_NAME)
        print(f"[INFO] Found {len(pdf_paths)} PDF(s) under {os.path.join(args.base_data_dir, VENDOR_NAME)}")

    if not pdf_paths:
        print("[WARN] No PDFs found. Nothing to do.")
        return

    # Optional: skip PDFs already present in the existing CSV
    if args.skip_existing and not args.single_pdf:
        already = load_processed_pdf_paths(args.output_csv)
        before = len(pdf_paths)
        pdf_paths = [p for p in pdf_paths if p not in already]
        print(f"[INFO] Skip-existing ON. {before - len(pdf_paths)} already in CSV, {len(pdf_paths)} left to process.")

        if not pdf_paths:
            print("[INFO] Nothing new to process. Exiting.")
            return

    ensure_output_dir(args.output_csv)
    client = build_openai_client()

    rows: List[Dict[str, Any]] = []

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        print(f"\n[INFO] Processing {idx}/{len(pdf_paths)}: {pdf_path}")

        try:
            text = extract_text_and_tables(pdf_path)
            images_b64 = extract_page_images_as_base64(pdf_path, max_pages=2)
            record = call_openai_for_esg_record(client, text, images_b64, pdf_path)
            row = normalise_row(record, pdf_path)
            rows.append(row)
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf_path}: {e}")
            continue

    # Write CSV
    if rows:
        # Default behaviour: overwrite CSV for a full run.
        # For single-pdf or skip-existing runs, append to keep existing rows.
        file_exists = os.path.exists(args.output_csv)
        if args.single_pdf or args.skip_existing:
            mode = "a" if file_exists else "w"
            write_header = not file_exists
        else:
            mode = "w"
            write_header = True

        with open(args.output_csv, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print(f"\n[INFO] Wrote {len(rows)} ESG row(s) to {args.output_csv}")
    else:
        print("[WARN] No ESG rows were written (all PDFs failed?).")


if __name__ == "__main__":
    main()
