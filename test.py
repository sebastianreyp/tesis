import json
import indicoio
import csv
import re
indicoio.config.api_key = "209867caeb463058e8ce631e616d64c8"
from unicodedata import normalize
from clean_text import clean

fp  = open('./data/prueba.json')
words= [word.strip() for line in fp.readlines() for word in line.split('$') if word.strip()]
#print(", ".join(words)) # or `print(words)` if you want to print out `words` as a list
#with open('./prueba.json') as f:
   # data = json.load(f)
data = []
for t in words:
    try:
        #print(json.loads(t)["retweeted_status"]["extended_tweet"]["full_text"])
        d = json.loads(t)["retweeted_status"]["extended_tweet"]["full_text"]
        if d not in data:
            data.append(d)
    except:
        try:
            #print(json.loads(t)["text"])
            d = json.loads(t)["text"]
            if d not in data:
                data.append(d)
        except:
            print("-")

csvFile_utf8 = open(r'.\data\base-uft8.csv', 'a')
csvFile_clean = open(r'.\data\clean.1.csv', 'a')
csvWriter_utf8 = csv.writer(csvFile_utf8)
csvWriter_clean = csv.writer(csvFile_clean)
datos = []
data_text = []
line_count = 1
for row in data:
	s = indicoio.sentiment(row)
	v = 1 if ( s > 0.5 ) else 0
	print("({}),{},{}".format(line_count, v, row))
	line_count += 1
	text_base = row
	text_utf8  = text_base.encode("utf-8")
	text_clean = " ".join(filter(lambda x:x[0]!='@', text_base.split()))
	text_clean = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+",
		r"\1", normalize("NFD", text_clean), 0, re.I)
	text_clean = normalize('NFC', text_clean)
	text_clean = clean(text_clean)
	text_clean = text_clean.encode('ascii', 'ignore')
	text_clean = text_clean.decode("utf-8")
	text_clean = text_clean.replace("emoji", "")
	if(text_clean != ""):
		if text_clean not in data_text:
			print("({}),{}".format(v,text_utf8))
			csvWriter_utf8.writerow([v, text_utf8])
			csvWriter_clean.writerow([v, text_clean])
			data_text.append(text_clean)


print(f'Add {line_count} lines.')


print(f'Save.')