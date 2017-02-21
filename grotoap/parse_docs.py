import xml.etree.ElementTree as ET
from grotoap.pdf_objects import *

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
    doc = parse_doc('grotoap2\\dataset\\00\\1276794.cxml')
    # print(doc.getFullText())
    print(doc.toString())

if __name__ == '__main__':
    main()