# deep-metadata-extraction
Models to identify and extract metadata from scientific literature; training and testing is done on the GROTOAP2 dataset. The models and training code are heavily derivative of Emma Strubell's cnn-spred code.

## GROTOAP2 Preproccessing
### To extract the article plaintext from TrueViz XML documents, run the following:
```python get_plaintext_sax.py [path to directory containing XML docs] [path to target file/directory] -f```

```-f``` is an optional flag indicating that all plaintext docs should be written to one file, one document per line.

### pdf_objects.py
Represents parsed documents as a heirarchical structure; the highest container is Document. The heirarchy is represented as follows:
```
-Document
---Page(s)
-----Zone(s)
------Lines(s)
--------Words(s)
```
Zone labels are parsed out of the TrueViz XML and applied to all lines and words within a zone; so zones, lines, and words all have labels. Zones, lines, and words also have bounding box information, in the form of a top-left corner vertex and bottom-right corner vertex.

### parse_docs_sax.py
To parse a single doc into a Document representation, call the parse_doc(filename) function from parse_docs_sax.py, passing it the path of the file to be parsed. The function returns a Document object from which features may be extracted.

### grotoap_to_tfrecords.py
Run the following command to parse a directory of GROTOAP2 documents into TFRecords containing training sequences.

```
python grotoap_to_tfrecords.py 
  --grotoap_dir [top level directory containing grotoap dataset] 
  --out_dir [directory in which to save TFRecords]
  --load_vocab [w2v embeddings file to build vocab with]
  --debug (whether to print debug info, default False)
  --bilou (whether to encode the wrod labels in BILOU format, default False)
  --seq_len [the maximum length of an example sequence, default 30]
  --x_bins [number of bins to use for x coordinate features, default 4]
  --y_bins [number of bins to use for y coordinate features, default 4] 
```

The path provided to the `--grotoap_dir` argument should point to the GROTOAP directory contaning a series of numbered subdirectories (e.g. 00, 01, ..., 99), each of which contains TrueViz XML documents. Each of these subdirectories will be converted into a single TFRecord containing examples from those documents; these will be saved in `--out_dir`. A vocabulary will be created from the text or embeddings file provided to `--load_vocab` - during the preprocessing, all tokens not occuring in the vocab will be marked OOV (out of vocab). The `--debug` flag prints more information about the processing. `--seq_len` specifies how long the training sequences should be - each document page is segmented into sequences of this length. `--x_bins` and `--y_bins` specify how many bins to use for the x and y coordinate features of each word.

To create train, dev, and test sets, run `grotoap_to_tfrecords.py` on specific train, test, and dev directories containing the desired GROTOAP2 subdirectories. For example, one could create the following train, test, and dev directories by moving the numbered subdirectories:

```
-grotoap/train
--00
--01
--02
  ...
--10

-grotoap/test
--11

-grotoap/dev
--12
```

To create a directory of training TFRecords for the train set, one might run `grotoap_to_tfrecords.py`, passing grotoap/train as the `--grotoap_dir` argument and grotoap-processed/train to the `--out_dir`. This would create a new directory grotoap-processed/train containing the TFRecord files *00.proto*, *01.proto*, etc. This directory may be passed as a training directory to the training code.
