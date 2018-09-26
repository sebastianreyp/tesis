import pandas as pd
import csv
import re
from unicodedata import normalize
from clean_text import clean


df = pd.read_csv("./data/data.csv")

csvFile_clean = open(r'.\data\clean.txt', 'a', newline='')
csvWriter_clean = csv.writer(csvFile_clean)

for index, t in df.iterrows():
    text_clean = " ".join(filter(lambda x:x[0]!='@', t['texto'].split()))
    text_clean = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+",
               r"\1", normalize("NFD", text_clean), 0, re.I)
    text_clean = normalize('NFC', text_clean)
    text_clean = clean(text_clean)
    text_clean = text_clean.encode('ascii', 'ignore')
    text_clean = text_clean.decode("utf-8")
    text_clean = text_clean.replace("emoji", "")
    csvWriter_clean.writerow([t['polaridad'], text_clean])
    print(index,text_clean)