# Can NLP predict the popularity of your favorite anime before huge investments? A study with LLM's

NOTA: por ahora, escala las imágenes a mano en GIMP xdxdxd
![random waifu](https://github.com/JesusASmx/Correlation-between-anime-synopsis-and-popularity-with-LLM-s/blob/main/assets/random_waifu_mini.png)

Paper: *ArXIV (when available)*

This repository is for reproduct the results. It also contains the mere first version of StoneAxe: the first extractor of Generalized Propp Functions.

If you are a peer reviewer and want to reproduce the results, I (Jesús) uploaded the dataset in the website of the journal.

DISCLAIMER: Due the GitHub constraints of storage, all tensors employed in the paper are not uploaded here. The provided code should be enough to reproduce them (all seeds are set in 42). However, it is possible to request the original tensors employed in the reported experiments, but the sent will be very slow and problematic! If you are patient enough, feel free to request them to jesus.jorge.armenta@gmail.com

INSTRUCTIONS FOR REPRODUCIBILITY:

Library versions:
Transformers - Latest (march 2024)
SpaCy - Latest (march 2024)
Torch - 1.10.1
We first installed the latest Transformers library, and then Torch==1.10.1.

Employed hardware:
CUDA ?????? 46gb VRAM.
OS: Ubuntu.

PART 1: Synopsis embeddings.

To obtain the GPT2 synopsis embeddings, you must first to fine-tune the huggingface GPT2 implementation. This can be done with the file FILE, who is a code adapted from PhD. George Mihaila's tutorial of classification with GPT2 LIGA: https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/: we adapted it for a regression task, we added the regression_report function and adjusted the loss to be in terms of MSE instead of crossentropy, and we adapted the dataset class for our own necessities (the dataset is fully stored in .csv files, and not in .txt files inside folders as it happens with Mihaila's tutorial)

PART 2: Mistral and Llama2 embeddings.

DISCLAIMER: Ollama with Mistral and Llama2 installed is required.
To obtain these embeddings, just run FILEMISTRAL and FILELLAMA2. Just be sure tu put your own ollama direction in the variable *ollama*

PART 3: StoneAxe embeddings.

DISCLAIMER: Ollama with Vicuna installed is required.
Although the code for generate the generalized Propp functions from the database is provided, we also provide the results in .csv format. Their columns are MAL ID, Function 1, ..., Function 50, Score. Also, a script who employs a non-fine-tunned GPT2 model for generate embedding for each function, obtain the mean and store it into a tensor is provided.
