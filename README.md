# cell-classifier

## Repository Structure

This repository is organized as follows:

<pre>
data/ # original dataset with dataset including train, valid, test split
figs/ # generated figures during data analysis
models/ # model weights
src/ # implementations of attention mechanism, transformer-based classifier, trainer, dataset and pre-processing
scripts/ # scrips for launching training of logistic regression and transformer-based model, data split and data analysis
tests/ # test for the custom attention layer
eval_models.ipynb # Jupyter notebook for evaluating models in the test set
</pre>

## Setup and Installation

### Dependencies

You can install the required dependencies using pip.

```bash
pip install -r requirements.txt
```

## Training the models

Logistic regression and transformer are trained using the training and validation splits:

```bash
python scripts/train_linear_classifier.py
```

```bash
python scripts/train_transformer.py
```

## Evaluating the models

You can use the Jupyter Notebook `eval_models.ipynb` to evaluate the best models in the held-out test set.