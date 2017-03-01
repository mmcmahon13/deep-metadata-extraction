import codecs
import xml.etree.ElementTree as ET
import xml.sax
from xml.sax import ContentHandler

from pdf_objects import *
import os

# SAX content handler to assemble words and lines from the characters, print them to the specified file
class ConstructLine(ContentHandler):

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

    def startElement(self, name, attrs):
        ## PAGE ELELMENT HANDLERS
        if name == 'Page':
            self.num_pages += 1
            # if we have an old page, add it to the doc
            if not self.cur_page is None:
                self.doc.addPage(self.cur_page)
            # create the new page object
            self.cur_page = Page(self.num_pages)

        ## ZONE ELEMENT HANDLERS
        elif name == 'Zone':
            self.num_zones += 1
            if not self.cur_zone is None:
                self.cur_page.addZone(self.cur_zone)
            self.cur_zone = Zone(self.num_zones)
        # if we see a bounding box element for the zone, take note
        # so we know to assign the vertices to the current zone when we find them
        elif name == 'ZoneCorners':
            self.bb_type = "zone"
        # if we see a classification label, apply it to the current zone and store it til we see a new one?
        elif name == 'Classification':
            self.word_label = attrs.get('Value', "")
            self.cur_zone.setLabel(self.word_label)

        ## LINE ELEMENT HANDLERS
        elif name == 'Line':
            self.num_lines += 1
            if not self.cur_line is None:
                self.cur_zone.addZone(self.cur_line, self.word_label)
            self.cur_line = Line(self.num_lines)

        ## WORD ELEMENT HANDLERS
        elif name == 'Word':
            self.num_words += 1
            if not self.cur_word is None:
                self.cur_word.setText(self.wordText)
                self.wordText = ""
                self.cur_zone.addWord(self.cur_word)
            self.cur_word = Word(self.num_words, self.word_label)
        elif name == 'GT_Text':
            self.wordText += attrs.get('Value', "")

    def endDocument(self):
        # write the last word to the last line
        if self.wordText != '':
            self.lineText += self.wordText + ' '
        self.wordText = ''

        # write the last line to the doc
        if self.lineText != '':
            self.output_file.write(self.lineText + '\n')
        self.lineText = ''

# TODO change this to use SAX parser
def parse_doc(doc_path, doc_id):
    tree = ET.parse(doc_path)
    root = tree.getroot()

    # todo get doc id

    doc = Document(doc_id)

    for p, page in enumerate(root.iter('Page')):
        # create new page
        cur_page = Page(p)
        # create zones for page
        for z,zone in enumerate(page.iter('Zone')):
            # get the zone corners
            for corner in zone.iter('ZoneCorners'):
                for v, vertex in enumerate(corner):
                    if v == 0:
                        top_left = vertex.attrib
                    else:
                        bottom_right = vertex.attrib
            # get the zone classification
            for classification in zone.iter('Classification'):
                label = classification[0].attrib['Value']
            # create zone
            cur_zone = Zone(z, label, top_left, bottom_right)
            # go through the lines for the current zone
            for l, line in enumerate(zone.iter('Line')):
                # get the line corners
                for corner in zone.iter('ZoneCorners'):
                    for v, vertex in enumerate(corner):
                        if v == 0:
                            top_left = vertex.attrib
                        else:
                            bottom_right = vertex.attrib
                # go through current line and get text
                line_text = ""
                for w, word in enumerate(line.iter('Word')):
                    cur_word = ""
                    for char in word.iter('Character'):
                        cur_word += char[3].attrib['Value']
                    line_text += cur_word
                    line_text += " "
                # create line
                cur_line = Line(l, label, top_left, bottom_right, line_text.strip())
                cur_zone.lines.append(cur_line)
            cur_page.zones.append(cur_zone)
        doc.pages.append(cur_page)
    return doc

def main():
    doc = parse_doc('C:\Users\Molly\Google_Drive\spring_17\deep-metadata-extraction\\grotoap\grotoap2\\dataset\\00\\1276794.cxml', '1276794')
    print(doc.getFullText())
    print(doc.toString())
    # create_plaintext_corpus('grotoap2\\dataset\\00', 'grotoap2_text')

if __name__ == '__main__':
    main()