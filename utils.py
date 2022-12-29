import json
import datetime


def write_json_line(file, text, lang, source):
    file.write(json.dumps({
        "text": text,
        "lang": lang,
        "source": source,
        "downloaded_timestamp": datetime.date.today().strftime("%m-%d-%Y")
    }) + "\n")