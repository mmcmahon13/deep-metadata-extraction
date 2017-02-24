import codecs
import xml.etree.ElementTree as ET
import xml.sax
from pdf_objects import *
import os

# TODO: rewrite this with SAX

# given a directory of docs in TrueViz format, get the plaintext files and write them to a new directory
def create_plaintext_corpus(trueviz_dir_path, target_dir_path):
    if not os.path.exists(target_dir_path):
        os.makedirs(target_dir_path)

    if not os.path.exists(trueviz_dir_path):
        print("Incorrect path to TrueViz directory: %s" % trueviz_dir_path)
        return
    else:
        for subdir, dirs, files in os.walk(trueviz_dir_path):
            for tv_file in files:
                if '.cxml' in tv_file:
                    full_text = ''
                    for event, elem in ET.iterparse(trueviz_dir_path + os.sep + tv_file):
                        if elem.tag == "Word":
                            cur_word = ''
                            for char in elem.iter('Character'):
                                cur_word += char[3].attrib['Value']
                            full_text += cur_word + " "
                        elem.clear()
                    print(full_text)
                    # doc = parse_doc(trueviz_dir_path + os.sep + tv_file)
                    text_file_path = target_dir_path + os.path.sep + tv_file
                    text_file = codecs.open(text_file_path, 'w', 'utf-8')
                    print("Writing file %s" % text_file_path)
                    # text_file.write(full_text)
                    text_file.close()
        print("Created directory of plaintext files")

def parse_doc(doc_path):
    tree = ET.parse(doc_path)
    root = tree.getroot()

    # todo get doc id

    doc = Document(1)

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
    # doc = parse_doc('grotoap2\\dataset\\00\\1276794.cxml')
    # print(doc.getFullText())
    # print(doc.toString())
    create_plaintext_corpus('grotoap2\\dataset\\00', 'grotoap2_text')

if __name__ == '__main__':
    main()