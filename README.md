# SemEval-2021 Task 2: Multilingual and Cross-lingual Word-in-Context Disambiguation (MCL-WiC)

This repository contains the software to train and evaluate the AERGCN model for the MCL-WiC competition.

## Requirements

- Python >= 3.7
- numpy >= 1.19.4
- scipy >= 1.5.4
- pandas >= 1.1.5
- matplotlib >= 3.3.3
- spacy >= 2.3.5
    - en-core-web-sm >= 2.3.1
    - zh-core-web-sm >= 2.3.1
    - fr-core-news-sm >= 2.3.1
- torch >= 1.7.1
- tensorboard >= 2.4.0
- transformers == 4.1.1
- pytokenizations >= 0.7.2
- wandb >= 0.8.36 (optional)

## Configurations

Please specifiy the used syntactic relation types and POS tag labels for the different languages under `/resources/language-name/`, i.e. for English [syntactic relation types](/resources/en/dependencies.txt) and [POS tag labels](/resources/en/pos_tags.txt).

## Example Usage

Please specify the command-line arguments for different settings, check details by running 

    python -m AERGCN -h

under the main folder. 

Each modules can be run individually for different functionlity tests, e.g.

    python -m AERGCN.data

### Training the model

Under the main folder, run the command

    python -m AERGCN

### Continue training the model with logs stored in *log_dir* (in the format: yyyy-mm-dd/model_num e.g. 2020-08-08/8/)

Under the main folder, run the command

    python -m AERGCN --log_dir /path-to-log_dir/

The format of the directory must comply with **yyyy-mm-dd/model_num** that contains the complete content folder of a (partially) trained model.

### Evaluating the model

Under the main folder, run the command

    python -m AERGCN --mode development(or test) --model_dir /path-to-model-pt/

where **devleopment** and **test** are for evaluating the specified model on the development split and the test split respectively. In the current version, the hyperparameters for configuring the model should be specified manually (can be checked in `params/params.json` under the trined model folder).

## License

GPL-3.0
