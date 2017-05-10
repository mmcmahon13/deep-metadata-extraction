# deep-metadata-extraction
Models to identify and extract metadata from scientific literature; training and testing is done on the GROTOAP2 dataset. The models and training code are heavily derivative of Emma Strubell's (@strubell) cnn-spred code.

## Requirements
This code requires Python 2.7, TensorFlow 0.11, cuda75/toolkit/7.5.18, and cudnn 5.1 to run (note - there may be more requirements that aren't listed here). The preprocessing requires that [Simstring](http://www.chokkan.org/software/simstring/) be installed. 

## Configuration files
The location of the data and hyperparameters for specific models and experiments are specified in configuration files, which live in the conf directory. These configuration files may be passed to specific scripts which train and evaluate models.

## GROTOAP2 Preproccessing
All of the preprocessing code for the GROTOAP2 dataset lives in the src/processing directory. It contains source code and bash scripts necessary to convert directories full of TrueViz XML documents to TFRecords, which may then be used for training/testing.

For more details, see the [wiki page](https://github.com/mmcmahon13/deep-metadata-extraction/wiki/GROTOAP2-Preprocessing). 
