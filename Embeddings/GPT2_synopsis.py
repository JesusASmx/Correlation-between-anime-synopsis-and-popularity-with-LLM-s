#######################################################################################
###THIS IS AN ADAPTED VERSION OF GEORGE MIHAILA'S CODE FOR CLASSIFICATING WITH GPT-2###
###https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/#########
#######################################################################################


###The most notable changes:
##1.- The database class. Since our data is loaded from .csv's than from .txt files into folders.
##2.- The training loop. We made the necessary adjustments for regressions.
##3.- We add an embedding-saver.

import pandas as pd

train_df = pd.read_csv("..//Database//anime_training.csv")
test_df = pd.read_csv("..//Database//anime_test.csv")

print(len(train_df)) #To be sure that the sizes matchs
print(len(test_df))

import io
import os
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (set_seed,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)


set_seed(42)
epochs = 4
batch_size = 16
max_length = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name_or_path = 'gpt2'


from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import numpy as np

class AnimeDataset(Dataset):
    def __init__(self, df, target_texts, target_label):
        global scaler

        self.texts = []
        self.labels = []
        self.lens = []
        for x in df.index:
            self.texto = df[target_texts][x]
            self.texts.append(self.texto)
            self.lens.append(len(self.texto))
            self.labels.append(df[target_label][x])

        print(max(self.lens))


        #Rescaling the labels:
        scaler.fit(np.array(self.labels).reshape(-1, 1))
        self.labels = list(scaler.transform(np.array(self.labels).reshape(-1, 1)))

        # Number of exmaples.
        self.n_examples = len(self.labels)
        return
    
    def __len__(self):
        return self.n_examples

    def __getitem__(self, item):
        return {'text':self.texts[item], 'label':self.labels[item]}


class Gpt2RegressionCollator(object):
    def __init__(self, use_tokenizer, max_sequence_len=None):
        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        return

    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_len)
        inputs.update({'labels': torch.tensor(np.array(labels), dtype=torch.float)})
        return inputs


def train(dataloader, optimizer_, scheduler_, device_):
    global model

    predictions = []
    true_labels = []

    total_loss = 0

    model.train()

    for batch in tqdm(dataloader, total=len(dataloader), desc="Batch"):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
        model.zero_grad()

        # Forward pass
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        predictions += logits.squeeze().detach().cpu()#.tolist()
        predictions = torch.mean(logits, dim=1, keepdim=True) #UNNCOMENT IF YOU SCALE THE REGRESSION VALUES
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(predictions, batch['labels'].float())
        total_loss += loss.item()
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_.step()
        scheduler_.step()

        predictions = predictions.tolist()

    avg_epoch_loss = total_loss / len(dataloader)
    return true_labels, predictions, avg_epoch_loss


def validation(dataloader, device_):
    global model

    all_preds = []
    predictions = []
    true_labels = []

    total_loss = 0

    model.eval()

    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits
            predictions += logits.squeeze().detach().cpu().tolist()
            all_preds += logits.squeeze().detach().cpu().tolist()

            predictions = torch.mean(logits, dim=1, keepdim=True) #for the loss calculation

            loss = torch.nn.functional.mse_loss(predictions, batch['labels'].float())
            total_loss += loss.item()

            predictions = predictions.tolist()

    avg_epoch_loss = total_loss / len(dataloader)
    return true_labels, [x[0] for x in all_preds], avg_epoch_loss  



print('Loading configuration...')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
model.to(device)
print('Model loaded to `%s`'%device)


gpt2_classificaiton_collator = Gpt2RegressionCollator(use_tokenizer=tokenizer,
                                                          max_sequence_len=max_length)


scaler = StandardScaler()

print('Dealing with Train...')
# Create pytorch dataset.
train_dataset = AnimeDataset(df=train_df,
                             target_texts="Synopsis",
                             target_label="Score"
                             )
print('Created `train_dataset` with %d examples!'%len(train_dataset))

# Move pytorch dataset into dataloader.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

print()

print('Dealing with Validation...')
# Create pytorch dataset.
valid_dataset = AnimeDataset(df=test_df,
                             target_texts="Synopsis",
                             target_label="Score")
print('Created `valid_dataset` with %d examples!'%len(valid_dataset))

# Move pytorch dataset into dataloader.
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))

# %%
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from scipy import stats

def regression_report(y_test, y_pred, save, name, prompt):
    mae = mean_absolute_error(y_test, y_pred)
    mdae = median_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    #mdape = ((pd.Series(y_test) - pd.Series(y_pred)) / pd.Series(y_test)).abs().median()
    r_squared = r2_score(y_test, y_pred)
    spearman = stats.spearmanr(y_test, y_pred)
    pearson = stats.pearsonr(y_test, y_pred)
    kendall = stats.kendalltau(y_test, y_pred)

    df = pd.DataFrame.from_dict({"Mean Absolute Error": [mae],
                                 "Median Absolute Error": [mdae],
                                 "Mean Squared Error": [mse],
                                 "Mean Absolute Percentage Error": [mape],
                                 #"MDAPE": [mdape],
                                 "R2 score": [r_squared],
                                 "Spearman": [spearman],
                                 "Pearson": [pearson],
                                 "KendallTau": [kendall]}
                                 )
    if prompt:
        print(f"R2 score: {r_squared}")
    if save:
        df.to_csv(".//Results//"+name+".csv")
    return df

# %%
optimizer = torch.optim.AdamW(model.parameters(),
                  lr = 2e-5, # default is 5e-5, 2e-5 first one
                  eps = 1e-8 # default is 1e-8.
                  )

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

all_loss = {'train_loss':[], 'val_loss':[]}
all_acc = {'train_acc':[], 'val_acc':[]}

# %%
for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch+1}/{epochs}")
    train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
    print('Evaluating over the validation set')
    valid_labels, valid_predict, val_loss = validation(valid_dataloader, device)
    reporte = regression_report(y_test=valid_labels, y_pred=valid_predict, save=False, name="GPT2_synopsis_score_linear_results", prompt=True)
    print(reporte)

    print("  train_loss: %.5f - val_loss: %.5f "%(train_loss, val_loss))
    print()

# %%
true_labels, predictions_labels, avg_epoch_loss = validation(valid_dataloader, device)

print("RESULTS IN TEST SET:")
reporte = regression_report(y_test=true_labels, y_pred=predictions_labels, save=True, name="GPT2_synopsis_score_linear_results", prompt=True)

true_labels = scaler.inverse_transform(np.array(true_labels).reshape(-1,1)).tolist()
predictions_labels = scaler.inverse_transform(np.array(predictions_labels).reshape(-1,1)).tolist()

# %%
##########################################
#GET EMBEDDINGS WITH THE FINE-TUNED MODEL#
##########################################

import numpy as np
from tqdm import tqdm

todos = [train_df["Synopsis"].tolist(), test_df["Synopsis"].tolist()]

names = {0:'train', 1:'test'}
temps = {0:None, 1:None}
for i in range(2):
    bsize = 32
    embeddings = []
    print(f"##########GETTING EMBEDDINGS FOR {names[i]}########")
    for x in tqdm(range(len(todos[i])//bsize+1), desc="Batch"):
        inputs = tokenizer(todos[i][bsize*x:min(bsize*(x+1),len(todos[i]))], return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        incrustación = outputs.hidden_states[-1]
        #print(incrustación.shape)
        embeddings.append(incrustación)

    embeddings = torch.cat(embeddings, dim=0).to(device)
    print(embeddings.shape)
    torch.save(embeddings, f'.//GPT2//GPT2_syn_score_{names[i]}.pt')

# %%


# %%



