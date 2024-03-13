# <center> Can NLP predict the popularity of your favorite anime before huge investments? A study with LLM's </center>

<p align="center">
    <a href=https://arxiv.org><img align="center" src="https://img.shields.io/badge/arXiv-mimi.mimim-b31b1b.svg"></a>
    <a href=https://google.com><img align="center" src="https://img.shields.io/badge/StoneAxe-V1.1-orange"></a>
    <a href=https://www.apache.org/licenses/LICENSE-2.0><img align="center" src="https://img.shields.io/badge/Licence-Apache_2.0-lightgray"></a>
</p>

<img align="center" src="https://github.com/JesusASmx/Correlation-between-anime-synopsis-and-popularity-with-LLM-s/blob/main/assets/anime_pop.png">

This is the official repository for reproduce the results reported at the paper:

```
Paper-Title Jesús Armenta-Segura, Olga Kolesnokova, Grigori Sidorov
```


## Reproducibility instructions

### Hardware Specifications

The experiments were performed with the follow hardware:
<ul>
    <li>MODELO DE LA TARJETA GRÁFICA.</li>
    <li>PROCESADOR.</li>
    <li>MEMORIA RAM.</li>
    <li>MEMORIA VRAM.</li>
</ul>


### Software Specifications

The employed software was the follow:
<ul>
    <li>CUDA ????</li>
    <li>OS: Ubuntu Server VERSION</li>
    <li>Vim version ???</li>
    <li>Python version: 3.???</li>
</ul>

And the python library versions were:
<ul>
    <li>Transformers - Latest (march 2024)</li>
    <li>SpaCy - Latest (march 2024)</li>
    <li>NumPy, Pandas, Tqdm, ..., ... - Latest (march 2024)</li>
    <li>Torch - 1.10.1</li>
</ul>

It is important to recall that we first installed the latest Transformers library, and then Torch==1.10.1.

Also, we employed a virtual enviroment, initialized with <i>venv</i>:

```python
python3 venv anime_virtual_enviroment
source anime_virtual_enviroment/bin/activate
```


### GPT2 embeddings of synopsis.

Just run the file GPT2_synopsis.py. Be sure to have the dataset at ```./Database``` (the dataset is available upon request).



### Mistral and Llama2 embeddings of synopsis.

Additional Requirements:
<ul>
    <li>Ollama. <a href=https://github.com/JesusASmx/Correlation-between-anime-synopsis-and-popularity-with-LLM-s/tree/main/Embeddings>(click here to check the specifications of our ollama)</a></li> 
</ul>

```python
def greet(name):
    print("Hello, " + name + "!")
```

GPT2 synopsis embeddings
Mistral and Llama2 synopsis embeddings <i>(requires ollama)</i>
StoneAxe 1.0 embeddings <i>(requires ollama)</i>
PCA representations
Regressions





