# deep-metadata-extraction
Models to identify and extract metadata from scientific literature; all preprocessing, training and testing is done on the GROTOAP2 dataset, which may be downloaded here: http://cermine.ceon.pl/grotoap2/.

The models and training code are heavily derivative of Emma Strubell's (@strubell) cnn-spred code.

## Requirements
This code requires Python 2.7, TensorFlow 0.11, cuda75/toolkit/7.5.18, and cudnn 5.1 to run (note - there may be more requirements that aren't listed here). The preprocessing requires that [Simstring](http://www.chokkan.org/software/simstring/) be installed. 

## Prerequisites
The scripts provided for training, testing, and preprocessing assume that the following environment variables have been set:
`$DEEP_META_ROOT` should point to the location of this repository on your machine. Simply cd to the root directory of the project and run the following:
```
export DEEP_META_ROOT=`pwd`
```

## GROTOAP2 Preproccessing
All of the preprocessing code for the GROTOAP2 dataset lives in the src/processing directory. It contains source code and bash scripts necessary to convert directories full of TrueViz XML documents to TFRecords, which may then be used for training/testing.

For more details, see the [wiki page](https://github.com/mmcmahon13/deep-metadata-extraction/wiki/GROTOAP2-Preprocessing). 


## Configuration files
The location of the data and hyperparameters for specific models and experiments are specified in configuration files, which live in the `conf` directory. These configuration files may be passed to specific scripts which train and evaluate models. They may be split into different types:

* `global.conf` specifies global default hyperparameter configurations for all models, as well as default directories for storing processed data and saving model checkpoints. This should be updated with the desired paths on your machine.

* `[name]_embeddings.conf` stores information related to a specific embeddings file, like the dimension and location of the file. These configuration files must be sourced by model configuration files.

* Model configuration files (`bilstm_full.conf`, etc) store hyperparamter and architecture information for specific models, and store the paths to the specific data directories used to train the model, as well as the directory to which it is saved. These files may be edited to point to the data of choice, using the embeddings file of choice, and the hyperparameter settings of choice.

## Training Scripts
The `bin` directory contains scripts for training, tuning, and evaluating the models specified by the configuration files. 

For example, to train a model using the configuration in `bilstm-full.conf`, simply run the following from the project root:
`./train_bilstm.sh conf/bilstm-full.conf`. 

To evaluate the trained model on the training and test set, run the following from the root directory:
`./eval_lstm conf/bilstm-full.conf`.

There are several additional scripts which train and evaluate models using only a subset of the features (omitting geometrical and dictionary matching features, for example).
