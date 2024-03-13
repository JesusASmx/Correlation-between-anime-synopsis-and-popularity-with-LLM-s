### STEP 1: Load the generalized propp functions. Run StoneAxe.py if you want to obtain them from scratch (warning: requires ollama).

import pandas as pd

train_df = pd.read_csv(".//Generalized Propp functions//stoneaxe_gpropp_train.csv")
test_df = pd.read_csv(".//Generalized Propp functions//stoneaxe_gpropp_test.csv")


### STEP 2: Open the GPT2 model.

import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_size = "left"
tokenizer.pad_token = tokenizer.eos_token

model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='gpt2').to(device)

model.eval()


###STEP 3: properly format the generalized Propp functions, in order to generate the GPT2 embeddings. In this proposal, we put them as list of lists. E.g.:
## train_samples = [[Function 1, Function 2, ..., Function 50], #Sample 1
##		[Function 1, Function 2, ..., Function 50], #Sample 2
##		...
##		...
##		...
##		[Function 1, Function 2, ..., Function 50], #Sample [-1]

train_samples = [[str(train_df[f"Function {i+1}"][x]) for i in range(50)] for x in train_df.index]
print(len(train_samples)) #To be sure we have all samples.
print(len(train_samples[0])) #To be sure we retrieved all 50 functions per sample.

test_samples = [[str(test_df[f"Function {i+1}"][x]) for i in range(50)] for x in test_df.index]
print(len(test_samples))
print(len(test_samples[0]))



### STEP 4: Generate the embeddings.

import numpy as np
from tqdm import tqdm

todos = [train_samples, test_samples] #Means "all" in spanish.
names = {0:'train',1:'test'}

for i in range(len(todos)):
    embeddings = []
    tod = todos[i]
    for x in tqdm(range(len(tod)), desc="Sample"):
        sample_embs = []
        for y in tod[x]:
            inputs = tokenizer(y, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                sample_embs.append(outputs.hidden_states[-1])
        tensor_stacked = torch.stack(sample_embs)
        mean_tensor = torch.mean(tensor_stacked, dim=0)
        embeddings.append(mean_tensor)

    embeddings = torch.cat(embeddings, dim=0).to(device)
    print(f"{name[i]} tensor shape:" embeddings.shape)
    nombre = f'GPT2_embs_sx_{name[i]}.pt' #Means "name" in spanish
    torch.save(embeddings, "..//Embeddings//StoneAxe_embs//"+nombre)