#Importar librerias
from config import *
import tweepy
import csv
import re
from unicodedata import normalize
from clean_text import clean

#Busca los tweets historicos con la palabra clave en formato UTF-8

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)
csvFile_utf8 = open(r'.\data\base-uft8.txt', 'a')
csvFile_clean = open(r'.\data\clean.txt', 'a')
csvWriter_utf8 = csv.writer(csvFile_utf8)
csvWriter_clean = csv.writer(csvFile_clean)
data_text = []
# TheBEAT_Lima, Uber, Uber_peru

for tweet in tweepy.Cursor(api.search, q="taxi OR taxibeat OR uber OR uber_peru OR Uber_peru",lang="es").items():
    text_base = tweet.text
    date = tweet.created_at
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
            print("({}),{},{}".format(len(data_text),date,text_utf8))
            csvWriter_utf8.writerow([date, text_utf8])
            csvWriter_clean.writerow([date, text_clean])
            data_text.append(text_clean)