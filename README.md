
## DiaASQ Data [Reproducibility Checklist 3 (b)&(c) 4 (a)]

The DiaASQ-ZH dataset can be found at:
  ```bash
  data/zh
    - train_preprocessed.json
    - valid_preprocessed.json
    - test_preprocessed.json
  ```

The DiaASQ-EN dataset can be found at:
  ```bash
  data/en
    - train_preprocessed.json
    - valid_preprocessed.json
    - test_preprocessed.json
  ```

## Code Usage [Reproducibility Checklist 4 (e)&(k)]

+ Train && Evaluate on the DiaASQ-ZH dataset
  ```bash 
  bash scripts/train_zh.sh
  ```

+ Train && Evaluate on the DiaASQ-EN dataset
  ```bash 
  bash scripts/train_en.sh
  ```

## Customized Hyperparameters [Reproducibility Checklist 4 (e)&(k)]

For the DiaASQ-ZH dataset, you can set hyperparameters in `main_zh_train.py` or `src\config_erlangshen_xlarge.yaml`, and the former has a higher priority.

For the DiaASQ-EN dataset, you can set hyperparameters in `main_en_train.py` or `src_en\en_config_roberta_large.yaml`, and the former has a higher priority.


## GPU memory requirements [Reproducibility Checklist 4 (f)]

  | Dataset | Batch size | GPU Memory |
  | --- | --- | --- |
  | DiaASQ-EN | 1 |  35GB |
  | DiaASQ-ZH | 1 | 32GB |


## Checkpoints of Models [Reproducibility Checklist 4 (e)&(k)]

The DiaASQ-ZH checkpoint can be found at:
  ```bash
  data/zh
    - train_preprocessed.json
  ```

The DiaASQ-EN checkpoint can be found at:
  ```bash
  data/en
    - train_preprocessed.json
  ```
