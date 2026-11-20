# Datasets

No dataset is included in this repository.

The application reads a CSV file with two columns:

| Column | Meaning |
|---|---|
| `class` | `0` = hate speech, `1` = offensive speech, `2` = normal speech |
| `tweet` | The text to classify |

The project was developed with a labelled tweet dataset based on HateXplain
(Mathew et al., 2021, https://github.com/hate-alert/HateXplain). Place your
CSV in this folder and choose it in the "Train Models" tab.

`RebalanceScript/rebalance_dataset.py` produces the rebalanced version used
in the project by oversampling the hate speech class.
