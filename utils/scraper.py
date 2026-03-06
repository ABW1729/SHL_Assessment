import os
import requests
import pandas as pd
import time
import re
from bs4 import BeautifulSoup

# =========================================
# CONFIG
# =========================================

BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "assessments.csv")

os.makedirs(DATA_DIR, exist_ok=True)


# =========================================
# DETAIL PAGE SCRAPER
# =========================================

def scrape_full_description(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return "", None, "", ""

        soup = BeautifulSoup(r.text, "html.parser")

        description = ""
        job_levels = ""
        languages = ""
        duration = None

        # Find all structured rows
        rows = soup.find_all(
            "div",
            class_="product-catalogue-training-calendar__row"
        )

        for row in rows:
            header = row.find("h4")
            if not header:
                continue

            title = header.get_text(strip=True)

            # ----------------------------
            # Description
            # ----------------------------
            if title == "Description":
                p = row.find("p")
                if p:
                    description = p.get_text(strip=True)

            # ----------------------------
            # Job Levels
            # ----------------------------
            elif title == "Job levels":
                p = row.find("p")
                if p:
                    job_levels = p.get_text(strip=True).rstrip(",")

            # ----------------------------
            # Languages
            # ----------------------------
            elif title == "Languages":
                p = row.find("p")
                if p:
                    languages = p.get_text(strip=True).rstrip(",")

            # ----------------------------
            # Duration
            # ----------------------------
            elif title == "Assessment length":
                p = row.find("p")
                if p:
                    match = re.search(r"(\d+)", p.get_text())
                    if match:
                        duration = int(match.group(1))

        return description, duration, job_levels, languages

    except Exception as e:
        print(f"Failed to scrape detail page: {url} | {e}")
        return "", None, "", ""


# =========================================
# MAIN SCRAPER
# =========================================

def scrape_shl_individual_tests():
    all_assessments = []

    print("Starting SHL Individual Test scraping...")

    for start in range(0, 500, 12):
        url = f"{BASE_URL}?start={start}&type=1"
        print(f"Scraping catalog page: {url}")

        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            break

        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")

        if not table:
            break

        rows = table.find_all("tr")[1:]
        if not rows:
            break

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue

            name_tag = cols[0].find("a")
            if not name_tag:
                continue

            name = name_tag.text.strip()
            url_full = "https://www.shl.com" + name_tag["href"]

            remote = "Yes" if cols[1].find("span", class_="-yes") else "No"
            adaptive = "Yes" if cols[2].find("span", class_="-yes") else "No"

            test_keys = cols[3].find_all("span")
            test_type = ", ".join(
                key.text.strip() for key in test_keys
            )

            # ---------------------------------
            # Scrape detail page
            # ---------------------------------
            description, duration, job_levels, languages = scrape_full_description(url_full)

            if not description:
                description = name

            if not duration:
                duration = 30

            all_assessments.append({
                "name": name,
                "url": url_full,
                "description": description,
                "duration": duration,
                "remote_support": remote,
                "adaptive_support": adaptive,
                "test_type": test_type,
                "job_levels": job_levels,
                "languages": languages
            })

            time.sleep(0.5)  # avoid blocking

        time.sleep(1)

    df = pd.DataFrame(all_assessments).drop_duplicates(subset="url")

    print("Total scraped:", len(df))

    if len(df) < 377:
        raise Exception("Less than 377 Individual Tests scraped!")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")


# =========================================
# RUN
# =========================================

if __name__ == "__main__":
    scrape_shl_individual_tests()