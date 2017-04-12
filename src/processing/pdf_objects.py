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
        if not self.top_left == (None, None) and not self.bottom_right == (None, None):
            return abs(self.top_left[1] - self.bottom_right[1])
        else:
            return None

    def width(self):
        if not self.top_left == (None, None) and not self.bottom_right == (None, None):
            return abs(self.top_left[0] - self.bottom_right[0])
        else:
            return None

    # from grotoap paper: 1 typographic point equals to 1/72 of an inch
    def centerpoint(self):
        if not self.top_left == (None, None) and not self.bottom_right == (None, None):
            x = float(abs(self.top_left[0] + self.bottom_right[0]))/2
            y = float(abs(self.top_left[1] + self.bottom_right[1]))/2
            return(x, y)
        else:
            return (None, None)

    def region(self):
        pass

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
        words = []
        for page in self.pages:
            for zone in page.zones:
                for line in zone.lines:
                    for word in line.words:
                        # print(word.id + '\t' + word.text + '\t' + word.label + '\t' + str(word.top_left) + '\t' + str(word.bottom_right))
                        words.append(word)
        # print("Num words: ", len(words))
        return words

    def lines(self):
        lines = []
        for page in self.pages:
            for zone in page.zones:
                for line in zone.lines:
                    # print(line.id + '\t'+ line.text + '\t' + line.label + '\t' + str(line.top_left) + '\t' + str(line.bottom_right))
                    lines.append(line)
        return line

    def zones(self):
        zones = []
        for page in self.pages:
            for zone in page.zones:
                # print(zone.id + '\t'+ zone.text + '\t' + zone.label + '\t' + str(zone.top_left) + '\t' + str(zone.bottom_right))
                zones.append(zone)
        return zones


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

# TODO: add pointers to containing objects so we can use them for feature engineering as well
class Word(DocumentObject, BoundedDocumentObject):
    """
        Representation of a word
        Words have labels (in our case, the same as enclosing lines) and bounding boxes
    """
    def __init__(self):
        self.place_score = None
        self.department_score = None
        self.university_score = None
        self.person_score = None

    def shape(self):
        if all(c.isupper() for c in self.text):
            return "AA"
        if self.text[0].isupper():
            return "Aa"
        if any(c for c in self.text if c.isupper()):
            return "aAa"
        else:
            return "a"