# deep-metadata-extraction
Models to identify and extract metadata from scientific literature

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

### parse_docs_sax
*In progress*
