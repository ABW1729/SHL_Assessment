import pandas as pd
import re

INPUT_FILE = "data/assessments.csv"
OUTPUT_FILE = "data/assessments.csv"


JOB_LEVELS = [
    "Director", "Executive", "Front Line Manager",
    "General Population", "Graduate", "Manager",
    "Mid-Professional", "Professional",
    "Individual Contributor", "Supervisor"
]

LANGUAGES = [
    "English International", "English (USA)",
    "French", "French (Canada)", "French (Belgium)",
    "German", "Italian",
    "Portuguese", "Portuguese (Brazil)",
    "Spanish", "Latin American Spanish",
    "Swedish", "Turkish", "Danish",
    "Norwegian", "Flemish"
]


def process_description(text):
    if not isinstance(text, str):
        return "", "", ""

    original = text

    # -------------------------
    # Extract Job Levels
    # -------------------------
    found_levels = [
        level for level in JOB_LEVELS if level in original
    ]

    # -------------------------
    # Extract Languages
    # -------------------------
    found_languages = [
        lang for lang in LANGUAGES if lang in original
    ]

    # -------------------------
    # Remove Browser Warning
    # -------------------------
    text = re.sub(
        r"We recommend upgrading to a modern browser.*?Latest browser options\.?",
        "",
        text,
        flags=re.IGNORECASE
    )

    # -------------------------
    # Remove Sample Reports
    # -------------------------
    text = re.split(
        r"Sample Report",
        text,
        flags=re.IGNORECASE
    )[0]

    # -------------------------
    # Remove Marketing/Footer
    # -------------------------
    text = re.split(
        r"Speak to our team|Book a Demo|All rights reserved",
        text,
        flags=re.IGNORECASE
    )[0]

    # -------------------------
    # Remove Job Levels & Languages from Description
    # -------------------------
    for level in JOB_LEVELS:
        text = text.replace(level, "")

    for lang in LANGUAGES:
        text = text.replace(lang, "")

    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text, ", ".join(found_levels), ", ".join(found_languages)


def main():
    df = pd.read_csv(INPUT_FILE)

    if "description" not in df.columns:
        raise Exception("description column not found in CSV")

    processed = df["description"].apply(process_description)

    df_new = df.copy()
    df_new["description"] = processed.apply(lambda x: x[0])
    df_new["job_levels"] = processed.apply(lambda x: x[1])
    df_new["languages"] = processed.apply(lambda x: x[2])

    df_new.to_csv(OUTPUT_FILE, index=False)

    print(f"New structured CSV created at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()