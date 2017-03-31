# sekelab
My code in sekelab

### Requirements
+ General
    + Python (verified on 3.5.2)
+ Python Packages
    + Tensorflow (deep learning library, verified on 1.0.0)
    + Scikit-learn (machine learning library, verified on 0.18.1)
    + nltk (NLP tools, verified on 3.2.1)
    + tqdm (progress bar, verified on 4.7.4)

### Pre-processing
- First, create a directory named 'data' in this project and create two directories named 'squad' and 'glove' in the 'data'. Download SQuAD data and GloVe. Copy SQuAD data json in the 'squad' and copy GloVe txt in the 'glove'.
    ```
    GloVe: http://nlp.stanford.edu/data/glove.6B.zip
    ```

- Second, download nltk corpus.
    ```
    python3 -c "import nltk; nltk.download('all')"
    ```
- Third, preprocess SQuAD data.

    ```
    python3 -m squad.prepro
    ```

### Training
- The model is trained with NVIDIA Tesla P4:

    ```
    python3 main.py
    ```
- Options

    ```
    --hiddenSize: hidden layer units, default = 50
    --lrate: learning rate, default = 0.002
    --batchSize: batch size, default = 10
    --dropoutRate: dropout rate, default = 1.0
    ```
