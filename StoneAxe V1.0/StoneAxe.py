### STEP 1: Load the dataset. Is available upon request (see readme.md in root)

import pandas as pd

train = pd.read_csv("..//Database//anime_training.csv")
test = pd.read_csv("..//Database//anime_test.csv")


import math

def batch_df(df, bsize):
    total_batches = math.ceil(len(df)/bsize)
    return [df.iloc[bsize*i:bsize*(i+1)] for i in range(total_batches)]



### STEP 2: Preliminar functions. Recall that StoneAxe algorithm is as follows:
##
## Input: text. (In this case, from the train and test df's)
##
## 1.- Get all named entities. (function ents_and_roots)
## 2.- For each named entity, get their dependency tree and retrieve the root. (function ents_and_roots)
## Function ents_and_roots returns a dictionary {named entities: [root, context with radius window (window words at left, window words at right)]}
## 
## 3.- For each named entity, root and context window, ask Vicuna to explain the "dramatic relationship" (functon llama_reviews)
## 
## Output: Vicuna's response.

import spacy

def ents_and_roots(sentence, nlp, window):
    doc = nlp(sentence)

    entidades = [x.text for x in doc.ents]
    ent_and_roots = {}
    for token in doc:
        if token.text in entidades:
            
            ##Obtain the root of the tree (ancestors method is not good enough):
            head = token.head
            while head != head.head: #In Spacy, if a token is root, its head is itself.
                head = head.head
                
            ##NEXT STEP: Obtain the sentence who contains head.
            head_sent = False
            for sent in doc.sents:
                if sent[-1].i >= head.i:
                    where_head = head.i - sent.start
                    start = max(0, where_head-window)
                    finish = min(where_head+window, sent.end)
                    head_sent = sent[start : finish]
                    break
            if not head_sent:
                print(head.text, [x for x in doc.sents])
                raise Exception("Token out of bounds!")
            ent_and_roots[token.text] = [head, head_sent]

    return ent_and_roots


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from pprint import pprint
import requests

def llama_reviews(entsroots, keyword, proxy, model, temperature):
    ejemplo = entsroots[keyword]
    prompt = f'Briefly describe the dramatic relationship between "{keyword}" and "{ejemplo[0]}" in the sentence "{ejemplo[1]}".' #You might be surprised about how much a single dot can improve your output :) (in this case, prevents the model to try to complete ejemplo[1])
    output = requests.post(
            proxy,
            json = {
                "model": model,
                "prompt": prompt,
                "options": {"seed": 42, "temperature": temperature},
                "stream": False
                },
            )
    output = output.json()
    return output['response']


### STEP 3: Implementation of the StoneAxe algorithm.

your_proxy = f"http://{HERE YOU SHOULD PUT YOUR LOCAL ADDRESS FOR OLLAMA}/api/generate"

def StoneAxe(sample, spacy_parser, window, your_proxy):
    entroot = ents_and_roots(sample, spacy_parser, window)
    revs = {}
    for i in range(len(entroot)):
        revs[f"Function {i+1}"] = llama_reviews(entroot,  list(entroot.keys())[i], proxy=your_proxy, model="vicuna", temperature=0.1)
    return revs

from tqdm import tqdm

def run(df, target, origin, score, spacy_parser, pad, your_proxy):
    discursemas = {} 

    for i in tqdm(df.index, desc="Processed sample"):
        sample = df[target][i]
        origen = df[origin][i] #Means "origin" in spanish.
        discursemas[origen] = StoneAxe(sample = sample, spacy_parser = spacy_parser, window = 10, your_proxy = your_proxy)
        if pad < len(discursemas[origen]):
            print(f"WARNING, SAMPLE {i} HAVE MORE NAMED ENTITIES THAN {pad} (PAD). (truncating)")
            discursemas = {x:discursemas[x] for x in list(discursemas.keys())[:pad]}

        if len(discursemas[origen]) < pad:
            llevamos = len(discursemas[origen])
            cuanto_pad = pad - llevamos
            for j in range(cuanto_pad):
                discursemas[origen][f"Function {llevamos + j + 1}"] = "" #Padding with an empty sentence
        discursemas[origen]["Score"] = df[score][i]

    return discursemas


### STEP 4: Run StoneAxe over all the dataset.

batch_size = PICK_YOUR_OWN_VALUE
nlp = spacy.load("en_core_web_sm")
start_tr = 0 #In case your run crashes, you can start from the last batch with this value.
train_batches = batch_df(df=train, bsize=batch_size)[start_tr:]

for i in range(len(train_batches)):
    print(f"Batch {i}/{len(train_batches)}")
    tb = train_batches[i]
    traindsemmes = run(df=tb, target="Synopsis", origin="index", score="Score", spacy_parser=nlp, pad=50, your_proxy=your_proxy)
    train_gpropp = pd.DataFrame.from_dict(traindsemmes, orient='index').reset_index()
    train_gpropp.to_csv(f".//Generalized Propp functions//stoneaxe_gpropp_train_batch{i}.csv") #Since we are mad 'n' nuts, we put batch_size = 5000, obtaining a single batch. Hence, we uploaded the file to the repository without the batch{i} name

start_ts = 0
test_batches = batch_df(df=test, bsize=batch_size)[start_ts:]

for i in range(len(test_batches)):
    print(f"Batch {i}/{len(test_batches)}")
    tb = test_batches[i]
    testdsemmes = run(df=tb, target="Synopsis", origin="index", score="Score", spacy_parser=nlp, pad=50, your_proxy=your_proxy)
    test_gpropp = pd.DataFrame.from_dict(testdsemmes, orient='index').reset_index()
    test_gpropp.to_csv(f".//Generalized Propp functions//stoneaxe_gpropp_test_batch{i}.csv")