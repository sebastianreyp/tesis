from nltk.corpus import stopwords
import string
import re
from nltk.stem.wordnet import WordNetLemmatizer
import csv
from unicodedata import normalize


def clean(doc):
    lm = WordNetLemmatizer()
    stop = set(stopwords.words('spanish'))
    stop.add('uber')
    stop.add('Uber_peru')
    stop.add('uber_peru')
    stop.add('username')
    stop.add('hashtag')
    stop.add('url')
    stop.add('wu')
    stop.add('taxi')
    stop.add('emoji')
    exclude = set(string.punctuation)
    #english_vocab = set(w.lower() for w in nltk.corpus.words.words())

    clean_text = re.sub(r'http\S+', '$URL$ ', doc)
    clean_text = re.sub(r'\$\w*', '', clean_text)
    clean_text = re.sub(r'['+string.punctuation+']+', ' ', clean_text)
    clean_text = re.sub(r'@\w*', '$USERNAME$ ', clean_text)
    clean_text = re.sub(r'#\w*', '$HASHTAG$ ', clean_text)
    clean_text = re.sub(
        u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])', '$EMOJI$ ', clean_text)
    clean_text = re.sub(
        u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])', '$EMOJI$ ', clean_text)
    clean_text = ''.join(ch for ch in clean_text if ch not in exclude)
    clean_text = " ".join(
        [i for i in clean_text.lower().split() if i not in stop and len(i) > 2])
    normalized = " ".join(lm.lemmatize(word) for word in clean_text.split())
    #normalized = " ".join([i for i in clean_text.lower().split() if i not in stop and len(i) > 2 and i in english_vocab])
    return normalized


data = []
data_text = []
with open('./data/ULTIMO.csv', newline='', encoding="ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        data.append(row[0])
csvFile_clean = open(r'.\data\clean2.csv', 'a', newline="\n")
csvWriter_clean = csv.writer(csvFile_clean)
for row in data:
    text_base = row
    text_utf8 = text_base.encode("utf-8")
    text_clean = " ".join(filter(lambda x: x[0] != '@', text_base.split()))
    text_utf8 = text_base.encode("utf-8")
    text_clean = " ".join(filter(lambda x: x[0] != '@', text_base.split()))
    text_clean = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+",
                        r"\1", normalize("NFD", text_clean), 0, re.I)
    text_clean = normalize('NFC', text_clean)
    text_clean = clean(text_clean)
    text_clean = text_clean.encode('ascii', 'ignore')
    text_clean = text_clean.decode("utf-8")
    text_clean = text_clean.replace("emoji", "")
    if(text_clean != ""):
        if text_clean not in data_text:
            csvWriter_clean.writerow([text_clean])
            data_text.append(text_clean)
print('FIN')
