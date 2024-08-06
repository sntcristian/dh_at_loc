import csv
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyvis.network import Network
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)


with open("rosa-parks-in-her-own-words-2021-04-19.csv", "r", encoding="utf-8") as f:
    data = csv.DictReader(f)
    data = list(data)


places = dict()
people = dict()
pbar = tqdm(total=len(data))

for row in data:
    text = row["Transcription"]
    text = re.sub("\s+", " ", text)
    result = nlp(text)
    curr_word = []
    curr_tag = None
    for ent in result:
        if ent["entity"].startswith("B"):
            if len(curr_word)>0:
                if curr_tag=="LOC":
                    name = "".join(curr_word)
                    if name in places.keys():
                        places[name]["frequency"]+=1
                        places[name]["items"].add(row["ItemId"])
                    else:
                        places[name] = dict()
                        places[name]["frequency"]=1
                        places[name]["items"]=set([row["ItemId"]])
                if curr_tag=="PER":
                    if len(curr_word) > 1:
                        name = "".join(curr_word)
                        if name in people.keys():
                            people[name]["frequency"]+=1
                            people[name]["items"].add(row["ItemId"])
                        else:
                            people[name]=dict()
                            people[name]["frequency"]=1
                            people[name]["items"]=set([row["ItemId"]])
            curr_word=[]
            curr_word.append(ent["word"])
            curr_tag = ent["entity"].split("-")[1]
        else:
            if ent["word"].startswith("#"):
                curr_word.append(ent["word"][2:])
            else:
                curr_word.append(" "+ent["word"])

    if len(curr_word) > 0:
        if curr_tag == "LOC":
            name = "".join(curr_word)
            if name in places.keys():
                places[name]["frequency"] += 1
                places[name]["items"].add(row["ItemId"])
            else:
                places[name]=dict()
                places[name]["frequency"] = 1
                places[name]["items"] = set([row["ItemId"]])
        if curr_tag == "PER":
            if len(curr_word)>1:
                name = "".join(curr_word)
                if name in people.keys():
                    people[name]["frequency"] += 1
                    people[name]["items"].add(row["ItemId"])
                else:
                    people[name]=dict()
                    people[name]["frequency"] = 1
                    people[name]["items"] = set([row["ItemId"]])
    pbar.update(1)

def plot_keys_by_frequency(data):
    # Extracting keys and their frequencies
    keys = list(data.keys())
    frequencies = [data[key]["frequency"] for key in keys]

    # Creating a DataFrame
    df = pd.DataFrame({"Key": keys, "Frequency": frequencies})

    # Sorting the DataFrame by frequency
    df = df.sort_values(by="Frequency", ascending=False)

    # Plotting the bar chart for the top 10 most frequent keys
    plt.bar(df["Key"][:10], df["Frequency"][:10], color='blue')
    plt.xlabel('Keys')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Frequent Keys')
    plt.xticks(rotation=45)
    plt.show()


def create_network(data, file_name):
    net = Network()
    # Adding nodes and edges
    for key, details in data.items():
        net.add_node(key)
        items = details["items"]
        for other_key, other_details in data.items():
            if key != other_key and set(items).intersection(set(other_details["items"])):
                net.add_node(other_key)
                net.add_edge(key, other_key)
    net.show(file_name+'.html', notebook=False)


# Using the functions
plot_keys_by_frequency(places)

plot_keys_by_frequency(people)


# ideas to develop
# plot locations and persons by frequency
# create social network based on co-occurence
# visualize Network with Pyvis

