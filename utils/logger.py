import datetime

DEBUG_FILE = "debug_log.txt"

def log_debug(stage, data):

    with open(DEBUG_FILE, "a", encoding="utf-8") as f:

        f.write("\n" + "="*80 + "\n")
        f.write(f"{datetime.datetime.now()} | {stage}\n")
        f.write("="*80 + "\n")
        f.write(str(data) + "\n")