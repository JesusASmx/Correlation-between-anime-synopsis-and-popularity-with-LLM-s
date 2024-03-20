# <center> Can NLP predict the popularity of your favorite anime before huge investments? A study with LLM's </center>

<p align="center">
    <a href=https://arxiv.org><img align="center" src="https://img.shields.io/badge/arXiv-comming.soon!-b31b1b.svg"></a>
    <a href=https://www.apache.org/licenses/LICENSE-2.0><img align="center" src="https://img.shields.io/badge/Licence-Apache_2.0-lightgray"></a>
</p>

<p align="center">
    UPDATE 14/03:  For now, I am only uploading the code for LLM's embeddings. Hence, this repo should be taken as a tutorial (with no didactical instructions) than as a repo for a paper. We will upload the rest of the code (mainly the PCA analysis and our own embeddings for tackle the task) once submitted the paper.
</p>

<img align="center" src="https://github.com/JesusASmx/Correlation-between-anime-synopsis-and-popularity-with-LLM-s/blob/main/assets/anime_pop.png">

This is the official repository for reproduce the results for my final project (to be published on a journal paper).


## Reproducibility instructions

### Hardware Specifications

The experiments were performed with the follow hardware:
<ul>
    <li>Graphic card: NVIDIA Quadro RTX 6000/8000</li>
    <li>Processor: Intel Xeon E3-1200</li>
    <li>RAM: 62 gb</li>
    <li>VRAM: 46 gb</li>
</ul>


### Software Specifications

The employed software was the follow:
<ul>
    <li>CUDA  V10.1.243</li>
    <li>OS: Ubuntu Server 20.04.3 LTS</li>
    <li>Vim version 8.1</li>
    <li>Python version: 3.8.10</li>
</ul>

And the python library versions were:
<ul>
    <li>Transformers: 4.39.0.dev0</li>
    <li>SpaCy: 3.7.4</li>
    <li>NumPy, Pandas, Tqdm - Latest (march 2024)</li>
    <li>Torch - 1.10.1</li>
</ul>

It is important to recall that we first installed the latest Transformers library, and then Torch==1.10.1.

Also, we employed a virtual enviroment, initialized with <i>venv</i>, in a VIM editor on Ubuserver:

```python
>>> python3 venv anime_virtual_enviroment
>>> source anime_virtual_enviroment/bin/activate
```


### GPT2 embeddings of synopsis.

Just run the file GPT2_synopsis.py inside ```./Embeddings```. Be sure to have the dataset at ```./Database``` (it will be available for the final version of the repo).

For the PCA plots, run the file GPT2_syn_PCA.ipynb inside ```./PCA analysis```. The figures reported in the paper are attached to the readme (and in the original run on the .ipynb file). Due memory constraints, tensors were truncated at 10,000

For the regressions, run the file GPT2_regression.py inside ```./Regressions```. The runs reported in the paper are stored in ```./Regressions/Original runs```


### Mistral and Llama2 embeddings of synopsis.

Additional Requirements:
<ul>
    <li>Ollama. <a href=https://github.com/JesusASmx/Correlation-between-anime-synopsis-and-popularity-with-LLM-s/tree/main/Embeddings>(click here to check the specifications of our ollama)</a></li> 
</ul>

Just run the files Mistral_synopsis.py and Llama2_synopsis.py inside ```./Embeddings```. Be sure to have the dataset at ```./Database```.

For the PCA plots, run the files Mistral_syn_PCA.ipynb and Llama2_syn_PCA.ipynb inside ```./PCA analysis```. The figures reported in the paper are attached to the readme (and in the original run on the .ipynb file). Regardless the GPT2 case, here tensor truncation was not required since the shape was [n° samples, 4096].

For the regressions, run the files Llama2_regression.py and Mistral_regression.py inside ```./Regressions```. The runs reported in the paper are stored in ```./Regressions/Original runs```


### Licence
This work is licensed under the <a href=https://www.apache.org/licenses/LICENSE-2.0>Apache 2.0 Licence</a>.

<a href=https://www.apache.org/licenses/LICENSE-2.0><img align="center" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZU59RTNWfMysx4V5PYNOW_WTchiizy3fU8BbmB0BL-g&s"></a>

By using any part of this work (implicitly or explicitly), you acknowledge that you have read the license terms, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not use the code.

### Citation

If you use the Dataset and/or use part of our code -implicitly/explicitly- for your research projects, please reference and cite this repository. Once submitted the paper, it will be mandatory to actualize all citations to it (we encourage you to visit this repo weekly, in order to be sure).




