from __future__ import division
from __future__ import print_function

import os
# from fuzzywuzzy import fuzz
from xml.sax import make_parser, ContentHandler
from xml.sax.handler import feature_namespaces, feature_external_ges

import sys

from pdf_objects import *


# SAX content handler to assemble words and lines from the characters, print them to the specified file
# TODO: update this to use the same preproccessing as the word embeddings (same tokenization, replace unicode with ascii)
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
            self.cur_page = Page()

        elif name == 'PageID':
            self.cur_page.id = attrs.get('Value')

        ## ZONE ELEMENT HANDLERS
        elif name == 'Zone':
            self.num_zones += 1
            self.cur_zone = Zone()

        elif name == 'ZoneID':
            self.cur_zone.id = attrs.get('Value')

        elif name == 'ZoneCorners':
            # mark the bounding box as a zone box
            self.bb_type = "zone"

        elif name == 'Category' and not self.cur_zone is None:
            # if we see a classification label, apply it to the current zone and store it til we see a new one
            self.word_label = attrs.get('Value')
            self.cur_zone.label = self.word_label

        ## LINE ELEMENT HANDLERS
        elif name == 'Line':
            self.num_lines += 1
            self.cur_line = Line()
            self.cur_line.label = self.word_label

        elif name == 'LineID':
            self.cur_line.id = attrs.get('Value')

        elif name == 'LineCorners':
            # mark the bounding box as a line box
            self.bb_type = "line"

        ## WORD ELEMENT HANDLERS
        elif name == 'Word':
            self.num_words += 1
            self.cur_word = Word()
            self.cur_word.label = self.word_label

        elif name == 'WordID':
            self.cur_word.id = attrs.get('Value')

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
            (x, y) = (float(attrs.get('x')), float(attrs.get('y')))
            if self.bb_type == 'zone' and self.num_vert == 0:
                self.cur_zone.top_left = (x, y)
                self.num_vert += 1
            elif self.bb_type == 'zone' and self.num_vert == 1:
                self.cur_zone.bottom_right = (x, y)
                self.num_vert = 0
            elif self.bb_type == 'line' and self.num_vert == 0:
                self.cur_line.top_left = (x, y)
                self.num_vert += 1
            elif self.bb_type == 'line' and self.num_vert == 1:
                self.cur_line.bottom_right = (x, y)
                self.num_vert = 0
            elif self.bb_type == 'word' and self.num_vert == 0:
                self.cur_word.top_left = (x, y)
                self.num_vert += 1
            elif self.bb_type == 'word' and self.num_vert == 1:
                self.cur_word.bottom_right = (x, y)
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
            self.cur_word.text = self.wordText
            self.wordText = ""
            self.cur_line.addWord(self.cur_word)

def words_to_bilou(doc, labels=['AUTHOR', 'TITLE', 'AUTHOR_TITLE', 'ABSTRACT', 'AFFILIATION']):
    last_label = ""
    last_tag = ""
    field_len = -1
    words = doc.words()
    for i, word in enumerate(words):
        cur_label = word.label
        if cur_label in labels and cur_label != last_label:
            # B - current word is starting a new type
            word.label = 'B-' + cur_label
            # check if the previous word was part of a multi-word thing, or if it is a unit thing
            if field_len > 1:
                words[i - 1].label = words[i-1].label.replace('I-','L-')
            elif field_len > 0:
                words[i - 1].label = words[i - 1].label.replace('B-', 'U-')
            field_len = 1
        elif cur_label in labels and cur_label == last_label:
            # I - current word is continuing a type
            word.label = 'I-' + cur_label
            field_len += 1
        else:
            # O - current word is not one of the types we care about
            word.label = 'O'
            if field_len > 1:
                words[i - 1].label = words[i-1].label.replace('I-','L-')
            elif field_len > 0:
                words[i - 1].label = words[i - 1].label.replace('B-', 'U-')
            field_len = 0
        last_label = cur_label

    if field_len > 1:
        words[-1].label = words[-1].label.replace('I-', 'L-')
    elif field_len > 0:
        words[-1].label = words[-1].label.replace('B-', 'U-')

def match_dictionaries(doc, place_set, department_set, university_set, person_set, matching='approx'):
    words = doc.words()

    if matching == 'approx':
        import simstring
        # load simstring dbs
        place_db = simstring.reader('dicts' + os.sep + 'places.db')
        department_db = simstring.reader('dicts' + os.sep + 'departments.db')
        university_db = simstring.reader('dicts' + os.sep + 'universities.db')
        person_db = simstring.reader('dicts' + os.sep + 'people.db')
        dbs = [place_db, department_db, university_db, person_db]

        for db in dbs:
            db.measure = simstring.cosine
            db.threshold = 0.9

    # check all unigrams
    for word in words:
        tok = word.text.lower().strip()
        # place_score = fuzz.token_set_ratio(tok, places) / 100
        # dept_score = fuzz.token_set_ratio(tok, depts) / 100
        # uni_score = fuzz.token_set_ratio(tok, unis) / 100
        # person_score = fuzz.token_set_ratio(tok, people) / 100
        # print("%s: place: %f department: %f university: %f person: %f" % (tok, place_score, dept_score, uni_score, person_score))
        if matching == 'exact':
            if tok in place_set:
                word.place_score = 1
            else:
                word.place_score = 0
            if tok in department_set:
                word.department_score = 1
            else:
                word.department_score = 0
            if tok in university_set:
                word.university_score = 1
            else:
                word.university_score = 0
            if tok in person_set:
                word.person_score = 1
            else:
                word.person_score = 0
        elif matching == 'approx':
            # print(tok)
            # print(type(tok))
            tok = tok.encode('ascii', 'ignore')
            # print(type(tok))
            sys.stdout.flush()
            word.place_score = 0
            word.place_score = 1 if len(place_db.retrieve(tok)) > 0 else word.place_score
            word.department_score = 0
            word.department_score = 1 if len(department_db.retrieve(tok)) > 0 else word.department_score
            word.university_score = 0
            word.university_score = 1 if len(university_db.retrieve(tok)) > 0 else word.university_score
            word.person_score = 1 if len(person_db.retrieve(tok)) > 0 else 0

            # if word.place_score == 1:
            #     print(tok, "place")
            # if word.department_score == 1:
            #     print(tok, "dept")
            # if word.university_score == 1:
            #     print(tok, "university")
            # if word.person_score == 1:
            #     print(tok, "person")


# TODO: try both binary matching and the weird prefix-suffix BILOU thing that the paper does
def load_dictionaries():
    # place dictionaries
    place_set = set([])
    with open('dicts' + os.sep + 'city_full.txt') as f:
        for word in f:
            place_set.add(word.strip())
    with open('dicts' + os.sep + 'country_full.txt') as f:
        for word in f:
            place_set.add(word.strip())
    with open('dicts' + os.sep + 'region_full.txt') as f:
        for word in f:
            place_set.add(word.strip())

    # department dictionaries
    department_set = set([])
    with open('dicts' + os.sep + 'department_full.txt') as f:
        for word in f:
            department_set.add(word.strip())
    with open('dicts' + os.sep + 'department_keywords.txt') as f:
        for word in f:
            department_set.add(word.strip())
    with open('dicts' + os.sep + 'faculty_full.txt') as f:
        for word in f:
            department_set.add(word.strip())
    with open('dicts' + os.sep + 'faculty_keywords.txt') as f:
        for word in f:
            department_set.add(word.strip())

    # university dictionaries
    university_set = set([])
    with open('dicts' + os.sep + 'university_full.txt') as f:
        for word in f:
            university_set.add(word.strip())
    with open('dicts' + os.sep + 'university_keywords.txt') as f:
        for word in f:
            university_set.add(word.strip())

    # person name dictionaries
    person_set = set([])
    with open('dicts' + os.sep + 'chinese_only.txt') as f:
        for word in f:
            person_set.add(word.strip())
    with open('dicts' + os.sep + 'english_only.txt') as f:
        for word in f:
            person_set.add(word.strip())
    with open('dicts' + os.sep + 'frequent_last_names.txt') as f:
        for word in f:
            person_set.add(word.strip())
    with open('dicts' + os.sep + 'shared.txt') as f:
        for word in f:
            person_set.add(word.strip())

    return place_set, department_set, university_set, person_set

def parse_doc(doc_path):
    parser = make_parser()
    parser.setFeature(feature_namespaces, False)
    parser.setFeature(feature_external_ges, False)

    # todo come up with docid
    doc = Document()
    dh = ParsTrueViz(doc)

    parser.setContentHandler(dh)
    parser.parse(doc_path)
    return doc

def main():
    doc = parse_doc('C:\Users\Molly\Google_Drive\spring_17\deep-metadata-extraction\\grotoap\grotoap2\\dataset\\00\\1276794.cxml')
    place_set, department_set, university_set, person_set = load_dictionaries()
    match_dictionaries(doc, place_set, department_set, university_set, person_set)
    words = doc.words()
    for word in words:
        print('%s , %d, %d, %d, %d' %(word.text, word.place_score, word.department_score, word.university_score, word.person_score))

    # print(doc.getFullText())
    # print()
    # print(doc.toString())
    # words_to_bilou(doc)
    # print("\nWORDS\n")
    # words = doc.words()
    # for word in words:
    #     print('%s , %s' %(word.text, word.label))
    # print("\nLINES\n")
    # lines = doc.lines()
    # print("\nZONES\n")
    # zones = doc.zones()
    #
    # print("word text: ", words[1].text)
    # print("top left vertex: ",words[1].top_left)
    # print("bottom right vertex: ",words[1].bottom_right)
    # print("shape: ",words[1].shape())
    # print("width: ", words[1].width())
    # print("height: ",words[1].height())
    # print("center: ",words[1].centerpoint())

if __name__ == '__main__':
    main()