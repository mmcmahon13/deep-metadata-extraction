# Classes to represent the document structure, as parsed out of the TrueViz XML format

class DocumentObject:

    def __init__(self):
        self.id = ''


class BoundedDocumentObject(DocumentObject):

    def __init__(self):
        self.label = ""
        self.top_left = (None, None)
        self.bottom_right = (None, None)
    
    def height(self):
        if not self.top_left == (None, None):
            return self.top_left[0] - self.bottom_right[0]
        else:
            return -1


class Document(DocumentObject):
    """
    Nested representation of a PDF document, as parsed from a TrueViz XML file
    A document is made up of pages
    """

    def __init__(self):
        self.pages = []
        self.text = ""

    def addPage(self, page):
        self.pages.append(page)
        self.text += page.text

    def toString(self):
        """ Return a string to pretty-(ish) print the document structure """
        s = ''
        for page in self.pages:
            s += "Page %s\n" % page.id
            for zone in page.zones:
                s += "\tZone %s\n" % zone.id
                s += "\tZone class: %s\n" % zone.label
                for line in zone.lines:
                    s += "\t\tLine %s\n" % line.id
                    s += "\t\t%s\n" % line.text
        return s

    def words(self):
        for page in self.pages:
            for zone in page.zones:
                for line in zone.lines:
                    for word in line.words:
                        print(word.id + '\t' + word.text + '\t' + word.label + '\t' + str(word.top_left) + '\t' + str(word.bottom_right))

    def lines(self):
        for page in self.pages:
            for zone in page.zones:
                for line in zone.lines:
                    print(line.id + '\t'+ line.text + '\t' + line.label + '\t' + str(line.top_left) + '\t' + str(line.bottom_right))

    def zones(self):
        for page in self.pages:
            for zone in page.zones:
                print(zone.id + '\t'+ zone.text + '\t' + zone.label + '\t' + str(zone.top_left) + '\t' + str(zone.bottom_right))


class Page(DocumentObject):
    """
    Representation of a parsed page - pages contain multiple zones
    """

    def __init__(self):
        self.zones = []
        self.text = ""

    def addZone(self, zone):
        self.zones.append(zone)
        self.text += zone.text


class Zone(DocumentObject, BoundedDocumentObject):
    """
    Representation of a parsed zone, containing multiple lines
    Zones have labels and bounding boxes
    """

    def __init__(self):
        self.lines = []
        self.text = ""

    def addLine(self, line):
        self.lines.append(line)
        self.text += line.text + "\n"


class Line(DocumentObject, BoundedDocumentObject):
    """
    Representation of a line
    Lines have labels (in our case, the same as zones) and bounding boxes
    """
    def __init__(self):
        self.words = []
        self.text = ""

    def addWord(self, word):
        self.words.append(word)
        self.text += word.text + " "


class Word(DocumentObject, BoundedDocumentObject):
    """
        Representation of a word
        Words have labels (in our case, the same as enclosing lines) and bounding boxes
    """

    def __init__(self):
        pass
