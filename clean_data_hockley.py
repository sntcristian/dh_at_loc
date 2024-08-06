import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

with open("hockley_2023-03-07.csv", "r", encoding="utf-8") as f:
    data = csv.DictReader(f)
    data = list(data)

with open("full_transcriptions.txt", "w", encoding="utf-8") as out_f:
    for row in data:
        text = row['Transcription']
        if len(text.strip())>0:
            out_f.write(text+"\n")
out_f.close()