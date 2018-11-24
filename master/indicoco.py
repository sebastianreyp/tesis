import csv
import indicoio
indicoio.config.api_key = "209867caeb463058e8ce631e616d64c8"

csvFile = open(r'.\data\indicoco2.csv', 'a', newline="\n")
csvWriter = csv.writer(csvFile)
with open('./data/clean2.csv', newline='', encoding="utf8") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for index,row in enumerate(spamreader):
        s = indicoio.sentiment(row[0])
        v = 1 if (s > 0.5) else 0
        csvWriter.writerow([v, row[0]])
        print("{}/{}".format(index+1,'3166'))
