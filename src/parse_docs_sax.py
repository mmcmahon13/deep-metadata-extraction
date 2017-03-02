import codecs
from xml.sax import make_parser, ContentHandler
from xml.sax.handler import feature_namespaces, feature_external_ges

from pdf_objects import *
import os

# SAX content handler to assemble words and lines from the characters, print them to the specified file
class ParsTrueViz(ContentHandler):

    def __init__(self, doc):
        # pass in empty document object
        self.doc = doc
        self.cur_page = None
        self.cur_zone = None
        self.cur_line = None
        self.cur_word = None
        self.wordText = ""
        self.word_label = None
        self.num_pages = 0
        self.num_zones = 0
        self.num_lines = 0
        self.num_words = 0
        self.bb_type = "" # keep track of whether the current vertices mark a zone, line, or word bounding box
        self.num_vert = 0 # keep track of whether this is the first or second vertex in a bounding box

    def startElement(self, name, attrs):
        ## PAGE ELELMENT HANDLERS
        if name == 'Page':
            self.num_pages += 1
            # set the id to be the current number of pages, update it with given id later
            self.cur_page = Page(self.num_pages)

        elif name == 'PageID':
            self.cur_page.setID(attrs.get('Value'))

        ## ZONE ELEMENT HANDLERS
        elif name == 'Zone':
            self.num_zones += 1
            self.cur_zone = Zone(self.num_zones)

        elif name == 'ZoneID':
            self.cur_zone.setID(attrs.get('Value'))

        elif name == 'ZoneCorners':
            # mark the bounding box as a zone box
            self.bb_type = "zone"

        elif name == 'Category' and not self.cur_zone is None:
            # if we see a classification label, apply it to the current zone and store it til we see a new one
            self.word_label = attrs.get('Value')
            self.cur_zone.setLabel(self.word_label)

        ## LINE ELEMENT HANDLERS
        elif name == 'Line':
            self.num_lines += 1
            self.cur_line = Line(self.num_lines)
            self.cur_line.setLabel(self.word_label)

        elif name == 'LineID':
            self.cur_line.setID(attrs.get('Value'))

        elif name == 'LineCorners':
            # mark the bounding box as a line box
            self.bb_type = "line"

        ## WORD ELEMENT HANDLERS
        elif name == 'Word':
            self.num_words += 1
            self.cur_word = Word(self.num_words)
            self.cur_word.setLabel(self.word_label)

        elif name == 'WordID':
            self.cur_word.setID(attrs.get('Value'))

        elif name == 'WordCorners':
            # mark the bounding box as a line box
            self.bb_type = "word"

        ## CHARACTER ELEMENT HANDLERS
        # append character text to the current word
        elif name == 'GT_Text':
            self.wordText += attrs.get('Value', "")

        elif name == 'CharacterCorners':
            self.bb_type = 'char'

        ## BOUNDING BOX HANDLERS
        elif name == 'Vertex':
            # todo I think these are stored as strings, we probably want them as doubles
            (x, y) = (attrs.get('x'), attrs.get('y'))
            if self.bb_type == 'zone' and self.num_vert == 0:
                self.cur_zone.setTopLeft(x, y)
                self.num_vert += 1
            elif self.bb_type == 'zone' and self.num_vert == 1:
                self.cur_zone.setBottomRight(x, y)
                self.num_vert = 0
            elif self.bb_type == 'line' and self.num_vert == 0:
                self.cur_line.setTopLeft(x, y)
                self.num_vert += 1
            elif self.bb_type == 'line' and self.num_vert == 1:
                self.cur_line.setBottomRight(x, y)
                self.num_vert = 0
            elif self.bb_type == 'word' and self.num_vert == 0:
                self.cur_word.setTopLeft(x, y)
                self.num_vert += 1
            elif self.bb_type == 'word' and self.num_vert == 1:
                self.cur_word.setBottomRight(x, y)
                self.num_vert = 0

    def endElement(self, name):
        ## PAGE ELELMENT HANDLERS
        if name == 'Page':
            self.doc.addPage(self.cur_page)

        ## ZONE ELEMENT HANDLERS
        elif name == 'Zone':
            self.cur_page.addZone(self.cur_zone)


        ## LINE ELEMENT HANDLERS
        elif name == 'Line':
            self.cur_zone.addLine(self.cur_line)

        ## WORD ELEMENT HANDLERS
        elif name == 'Word':
            self.cur_word.setText(self.wordText)
            self.wordText = ""
            self.cur_line.addWord(self.cur_word)


    # def endDocument(self):
    #     self.cur_word.setText(self.wordText)
    #     self.cur_line.addWord(self.cur_word)
    #     self.cur_zone.addLine(self.cur_line)
    #     self.cur_page.addZone(self.cur_zone)
    #     self.doc.addPage(self.cur_page)

def parse_doc(doc_path, doc_id):
    parser = make_parser()
    parser.setFeature(feature_namespaces, False)
    parser.setFeature(feature_external_ges, False)

    # todo come up with docid
    doc = Document(doc_id)
    dh = ParsTrueViz(doc)

    parser.setContentHandler(dh)
    parser.parse(doc_path)
    return doc

def main():
    doc = parse_doc('C:\Users\Molly\Google_Drive\spring_17\deep-metadata-extraction\\grotoap\grotoap2\\dataset\\00\\1276794.cxml', '1276794')
    # print(doc.getFullText())
    # print()
    # print(doc.toString())
    # print()
    # doc.words()
    # print()
    doc.lines()
    # print()
    # doc.zones()

if __name__ == '__main__':
    main()