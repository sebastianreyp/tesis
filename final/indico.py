import indicoio
import csv
indicoio.config.api_key = "209867caeb463058e8ce631e616d64c8"

datos = []
row_num = 1
with open(r'.\data\clean.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if(row_num%2!=0):
            s = indicoio.sentiment(row[1])
            line_count += 1
            print("({}),{},{}".format(line_count, s, row[1]))
            datos.append([row[0], row[1], s])
        row_num+=1
    print(f'Add {line_count} lines.')


for d in datos:
    output = open(r".\data\res.txt", "a")
    output.write("{},{},{}\n".format(d[0],d[1],d[2]))
    output.close()

print(f'Save.')