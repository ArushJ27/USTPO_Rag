import os
import csv
from pathlib import Path
from bs4 import BeautifulSoup

# Configuration: adjust this path if needed
RAW_ROOT = Path("data/raw/Patent Gazettes/2024")
OUTPUT_CSV = Path("data/processed/patents_2024.csv")

# Ensure output directory exists
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# CSV header
headers = [
    "gazette",
    "html_file",
    "patent_number",
    "title",
    "inventor",
    "assignee",
    "classification",
    "image_path",
    "description"
]

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()

    # Iterate each gazette folder (e.g., 1530-1, 1530-2, ...)
    for gazette_dir in sorted(RAW_ROOT.iterdir()):
        if not gazette_dir.is_dir():
            continue
        gazette_name = gazette_dir.name
        html_subfolder = gazette_dir / "OG" / "html" / gazette_name
        if not html_subfolder.exists():
            # Fallback: some structures omit repeating folder name
            html_subfolder = gazette_dir / "OG" / "html"
        if not html_subfolder.exists():
            print(f"Warning: html folder not found for {gazette_name}")
            continue

        # Process each HTML file in this folder
        for html_file in sorted(html_subfolder.glob("*.html")):
            try:
                soup = BeautifulSoup(html_file.read_text(encoding="utf-8"), "html.parser")

                # Extract patent number and title from first table
                first_table = soup.find_all('table')[0]
                bolds = first_table.find_all('b')
                patent_number = bolds[0].get_text(strip=True) if len(bolds) > 0 else ""
                title = bolds[1].get_text(strip=True) if len(bolds) > 1 else ""

                # Inventor and assignee
                inventor = bolds[2].get_text(strip=True) if len(bolds) > 2 else ""
                assignee = bolds[4].get_text(strip=True) if len(bolds) > 4 else ""

                # Classification (CPC)
                classification = ""
                cpc_td = soup.find('td', text=lambda t: t and 'CPC' in t)
                if cpc_td:
                    # The td text contains 'CPC H01L ...'; extract after CPC
                    classification = cpc_td.get_text(" ", strip=True).replace("CPC", "").strip()

                # Image path
                img_tag = soup.find('img')
                image_path = img_tag['src'] if img_tag and 'src' in img_tag.attrs else ""
                # Normalize path relative to RAW_ROOT
                if image_path:
                    # If relative path, append to folder
                    image_path = str((html_file.parent / image_path).resolve())

                # Description: combine all <td class="para_text">
                desc_parts = [td.get_text(" ", strip=True) for td in soup.find_all('td', class_='para_text')]
                description = " ".join(desc_parts)

                # Write to CSV
                writer.writerow({
                    "gazette": gazette_name,
                    "html_file": str(html_file.resolve()),
                    "patent_number": patent_number,
                    "title": title,
                    "inventor": inventor,
                    "assignee": assignee,
                    "classification": classification,
                    "image_path": image_path,
                    "description": description
                })

            except Exception as e:
                print(f"Error parsing {html_file}: {e}")

print(f"Parsing complete. Output saved to {OUTPUT_CSV}")
