# Making the most of small Software Engineering datasets with modern machine learning

## Using the pre-trained models in your own script

### StackOBERTflow

We released and uploaded the model to Huggingface's model hub ([StackOBERTflow-comments-small-v1](https://huggingface.co/giganticode/StackOBERTflow-comments-small-v1))
The model can thus be used directly through the `transformers` library:

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
The following models are available:
* [BERT (base) fine-tuned on the AR Miner dataset](https://github.com/giganticode/small-datasets-ml-resources/releases/download/0.1/bert_ar_miner.zip)
* [RoBERTa fine-tuned on the AR Miner dataset](https://github.com/giganticode/small-datasets-ml-resources/releases/download/0.1/roberta_ar_miner.zip)
* [BERT (base) fine-tuned on 1M StackOverflow comments](https://github.com/giganticode/small-datasets-ml-resources/releases/download/0.1/stackoverflow_1M.zip)
* [BERT (base) fine-tuned on 2M StackOverflow comments](https://github.com/giganticode/small-datasets-ml-resources/releases/download/0.1/stackoverflow_2M.zip) 
* [BERT (large) fine-tuned on 1M StackOverflow comments](https://github.com/giganticode/small-datasets-ml-resources/releases/download/0.1/stackoverflow_1M_large.zip) 
* [BERT (base) fine-tuned on Java comments](https://github.com/giganticode/small-datasets-ml-resources/releases/download/0.1/bert_comments.zip)

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
  
tokenizer = AutoTokenizer.from_pretrained("model/path")
model = AutoModelForMaskedLM.from_pretrained("model/path")
```

# Tasks

The datasets cover 9 tasks, briefly described in the following.

## Sentiment Classification
Classify the sentiment of Software Engineering artifact (e.g., Stack Overflow posts, app reviews, bug report comments etc.)
Each example is to be classified as having either positive, negative or neutral sentiment.

We use the [Senti4SD](https://github.com/collab-uniba/Senti4SD) and [SentiData](https://sentidata.github.io) datasets.

### Examples

| Example | Label |
----------| ------|
| I want them to resize based on the length of the data they’re showing. | neutral |
| When I run my client, it throws the following exception. | negatve |
| This is always a really bad way to design software. | negative |
| amazing! a must have app | positive |

## App Review Classification

Classify app reviews into various categories (e.g., rating, feature request, bug report etc.) or
detect whether reviews are informative or not.
We use the datasets from [ARMINER](https://github.com/jinyyy666/AR_Miner/tree/master/datasets),
 [MAST](https://mast.informatik.uni-hamburg.de/wp-content/uploads/2014/03/REJ_data.zip)
and [CLAP](https://dibt.unimol.it/reports/clap).

### Examples

| Example | Label |
----------| ------|
| not able to download any pictures please fix these bugs immediately | informative |
| Best game I’ve played on Android | rating |
| good but... it has ads...please remove ads from this... | usability |


## Self-admitted technical debt detection

Detect whether a comment contains self-admitted technical debt (often indicated by `FIXME` or `TODO`).
We use the [dataset](https://github.com/maldonado/tse.satd.data) by Maldonado et. al.

### Examples

| Example | Label |
----------| ------|
| // FIXME: Is "No Namespace is Empty Namespace" really OK? | positive |
| // Can return null to represent the bootstrap class loader. | negative |

## Comment classification

Classify comments according to a pre-defined taxonomy (e.g., usage, license, deprecation,  ownership).
We rely on the [dataset](https://zenodo.org/record/2628361) by Pascarella et. al.

### Examples

| Example | Label |
----------| ------|
| // @return a string for throwing | usage |
| // New button,purpose | summary |
| // Caller of this method must hold that lock. | rationale |

## Code-Comment Coherence

Determine whether there is "coherence" between a given method and is corresponding lead comment, that is,
whether the comment is describtive of the method. 
We use the [dataset](http://www2.unibas.it/gscanniello/coherence/) by Corazza et. al.

### Example

```java
/**
* Returns the current number of milk units in
* the inventory.
* @return int
Code-Comment Coherence
*/
Prediction [25]
public int getMilk() {
  return milk;
}
```
*Label:*: positive (coherent) 

```java
/**
   * Check inventory user interface that processes input.
   */
public static void checkInventory() {
  System.out.println(coffeeMaker.checkInventory());
  mainMenu();
}
```
**Label:** negative (incoherent) 

## Linguistic Smell Detection

Detect linguistic smells in code, that is misleading
identifier names or the violations of common naming conventions. 
Our work is based on the [dataset](https://github.com/Smfakhoury/SANER-2018-KeepItSimple-) by Fakhoury et. al.

### Example

```java
public void ToSource(StringBuilder sb) {
  sb.append(";");
  this.NewLine(sb);
}
``` 
**Label:** smelly (transform method does not return)

## Code complexity classification

Classify the algorithmic complexity of various algorithm implementations (e.g., O(1), O(n*log(n)) etc.).
We use the [dataset](https://github.com/midas-research/corcod-dataset) by Sikka et. al.

### Example

```java
class GFG {
  static int minJumps(int arr[], int n) {
    int[] jumps = new int[n];
    int min;
    jumps[n - 1] = 0;
    for (int i = n - 2; i >= 0; i--) {
      if (arr[i] == 0)
        jumps[i] = Integer.MAX_VALUE;
      else if (arr[i] >= n - i - 1) 
        jumps[i] = 1;
      else { ... }
    }
    return jumps[0];
  }
  public static void main(String[] args) {...}
}
``` 
**Label:** smelly (transform method does not return)

## Code readability classification

Given a piece of code, classify it as either "readable" or "not readable".
Our work relies on the [dataset](https://dibt.unimol.it/report/readability/) by Scalabrino et. al.

### Example

```java
@Override
public void configure(Configuration cfg) {
  super.configure(cfg);
  cfg.setProperty(Environment.USE_SECOND_LEVEL_CACHE, ...);
  cfg.setProperty(Environment.GENERATE_STATISTICS, ...);
  cfg.setProperty(Environment.USE_QUERY_CACHE, "false" );
  ... // more cfg.setProperty calls
}
``` 
**Label:** readable

## Reproducing results


There is a module for each task/dataset. Currently, these are: 

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
 
Some of the used datasets (e.g., CLAP) are not publicly avaiable.
Datasets were preprocessed and brought into a standard format. If you like to rerun one of the experiments, please contact
one of the authors for the dataset in the correct format. Datasets must be placed in `/data/<module>/`.
Training parameters can be set in `/dl4se/config/<module>`, dataset loading is handled in `/dl4se/datasets/<module>`.

Additional configuration parameters can be passed on the command line. See the `config.py` file of the corresponding module for a list
of possible command line options.

To run an experiment execute the following:
```
python -mdl4se.experiments.<module>.default --seeds 100 200 300 400 500 --out_file=result_file.csv
```


