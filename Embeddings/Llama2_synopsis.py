import pandas as pd

train_df = pd.read_csv("..//Database//anime_training.csv")
test_df = pd.read_csv("..//Database//anime_test.csv")


from pprint import pprint
import requests
import numpy as np
import torch

def get_ollama_emb(prompt, seed, temp, model, proxy):
    output = requests.post(
            proxy,
            json = {
            "model": model,
            "prompt": prompt,
            "options": {"seed": seed, "temperature": temp},
            "stream": False
            }
        )
    output=output.json()
    embs = output['embedding']
    return torch.tensor(embs)


from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_embs(seed, temp, model, df, text, split, bsize, start, proxy):
    samples = df[text].tolist()[start*bsize:min((start+1)*bsize, len(df[text].tolist()))]
    #print(f"EXAMPLE OF SAMPLE: {samples[0][-15:]}")
    tensors = []
    for x in tqdm(samples, desc= model+" embeddings"):
        array = get_ollama_emb(prompt=x, seed=seed, temp=temp, model=model, proxy=proxy)
        tensors.append(array)
    tensors = torch.stack(tensors, dim=0).to(device)
    print(tensors.shape)
    torch.save(tensors, './/LLAMA2//'+model+f'_syn_score_{split}_batch{str(start)}.pt')
    print("Tensors successfully saved!")


import math
bsize = 32
bstart_train = 0
print(len(train_df))

your_proxy = f'http://{HERE YOU PUT YOUR LOCAL ADDRESS FOR OLLAMA}/api/embeddings'

print(f"Starting batch: {bstart_train}\nTotal batches: {math.ceil(len(train_df)/bsize)}")
for x in tqdm(range(bstart_train, math.ceil(len(train_df)/bsize)), desc="Batch"):
    get_embs(seed=42, temp=0.2, model="llama2", df=train_df, text="Synopsis", split="train", bsize=bsize, start=x, proxy=your_proxy)

bstart_test = 0
for x in tqdm(range(bstart_test, math.ceil(len(test_df)/bsize)), desc="Batch"):
    get_embs(seed=42, temp=0.2, model="llama2", df=test_df, text="Synopsis", split="test", bsize=bsize, start=x)

#a = get_ollama_emb(prompt = frieren, seed = 42, temp = 0.2, "mistral")
#print(a)
