# Making the most of small Software Engineering datasets with modern machine learning

## Using the pre-trained models in your own script

### StackOBERTflow

This model can be used directly through the `transformers` library:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
  
tokenizer = AutoTokenizer.from_pretrained("giganticode/StackOBERTflow-comments-small-v1")
model = AutoModelForMaskedLM.from_pretrained("giganticode/StackOBERTflow-comments-small-v1")
```

Using `pipeline`

```python
from transformers import pipeline
from pprint import pprint

COMMENT = "You really should not do it this way, I would use <mask> instead."

fill_mask = pipeline(
    "fill-mask",
    model="giganticode/StackOBERTflow-comments-small-v1",
    tokenizer="giganticode/StackOBERTflow-comments-small-v1"
)

pprint(fill_mask(COMMENT))
# [{'score': 0.019997311756014824,
#   'sequence': '<s> You really should not do it this way, I would use jQuery instead.</s>',
#   'token': 1738},
#  {'score': 0.01693696901202202,
#   'sequence': '<s> You really should not do it this way, I would use arrays instead.</s>',
#   'token': 2844},
#  {'score': 0.013411642983555794,
#   'sequence': '<s> You really should not do it this way, I would use CSS instead.</s>',
#   'token': 2254},
#  {'score': 0.013224546797573566,
#   'sequence': '<s> You really should not do it this way, I would use it instead.</s>',
#   'token': 300},
#  {'score': 0.011984303593635559,
#   'sequence': '<s> You really should not do it this way, I would use classes instead.</s>',
#   'token': 1779}]
```

### Fine-tuned models

Download the zipped model from the release page and extract to `model/path`.

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
  
tokenizer = AutoTokenizer.from_pretrained("model/path")
model = AutoModelForMaskedLM.from_pretrained("model/path")
```

## Reproducing results


There is a module for each dataset. Currently, these are: 

| Module | Description |  Dataset URL | 
---------| ------------|--------------|
| `ar_miner` | Informative app reviews | https://github.com/jinyyy666/AR_Miner/tree/master/datasets |
| `coherence` | Code-comment coherence | http://www2.unibas.it/gscanniello/coherence/ |
| `comment_classification` | Comment classification | https://zenodo.org/record/2628361 |
| `corcod` | Runtime complexity classification | https://github.com/midas-research/corcod-dataset |
| `readability` | Code readability classification | https://dibt.unimol.it/report/readability/ |
| `review_classification` | Review classification | https://mast.informatik.uni-hamburg.de/wp-content/uploads/2014/03/REJ_data.zip |
| `satd` | Self-admitted debt detection | https://github.com/maldonado/tse.satd.data |
| `senti4sd` | Sentiment analysis on Stack Overflow comments | https://github.com/collab-uniba/Senti4SD |
| `smell_detection` | Linguistic smell detection | https://github.com/Smfakhoury/SANER-2018-KeepItSimple- |
 
Some of the used datasets (e.g., CLAP are not publicly avaiable).
Datasets were preprocessed and brought into a standard format. If you like to rerun one of the experiments, please contact
one of the authors for the dataset in the correct format. Datasets must be placed in `/data/<module>/`.
Training parameters can be set in `/dl4se/config/<module>`, dataset loading is handled in `/dl4se/datasets/<module>`.

Additional configuration parameters can be passed on the command line. See the `config.py` file of the corresponding module for a list
of possible command line options.

To run an experiment execute the following:
```
python -mdl4se.experiments.<modules>.default --seeds 100 200 300 400 500 --out_file=result_file.csv
```


