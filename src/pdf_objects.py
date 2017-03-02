# Classes to represent the document structure, as parsed out of the TrueViz XML format

class Document:
    """
    Nested representation of a PDF document, as parsed from a TrueViz XML file
    A document is made up of pages
    """

    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.pages = []
        self.text = ""

    def addPage(self, page):
        self.pages.append(page)
        self.text += page.getFullText()

    def setID(self, id):
        self.doc_id = id

    def getFullText(self):
        """ Return the full plaintext from the document """
        # full_text = ''
        # for page in self.pages:
        #     full_text += page.getFullText()
        # return full_text
        return self.text

    def toString(self):
        """ Return a string to pretty-(ish) print the document structure """
        s = ''
        for page in self.pages:
            s += "Page %d\n" % page.page_id
            for zone in page.zones:
                s += "\tZone %d\n" % zone.zone_id
                s += "\tZone class: %s\n" % zone.label
                for line in zone.lines:
                    s += "\t\tLine %d\n" % line.line_id
                    s += "\t\t%s\n" % line.getFullText()
        return s

    def words(self):
        for page in self.pages:
            for zone in page.zones:
                for line in zone.lines:
                    for word in line.words:
                        print(word.word_id + '\t' + word.text + '\t' + word.label + '\t' + str(word.top_left) + '\t' + str(word.bottom_right))

    def lines(self):
        for page in self.pages:
            for zone in page.zones:
                for line in zone.lines:
                    print(line.line_id + '\t'+ line.getFullText() + '\t' + line.label + '\t' + str(line.top_left) + '\t' + str(line.bottom_right))

    def zones(self):
        for page in self.pages:
            for zone in page.zones:
                print(zone.zone_id + '\t'+ zone.getFullText() + '\t' + zone.label + '\t' + str(zone.top_left) + '\t' + str(zone.bottom_right))


class Page:
    """
    Representation of a parsed page - pages contain multiple zones
    """

    def __init__(self, page_id):
        self.page_id = page_id
        self.zones = []
        self.text = ""

    def setID(self, id):
        self.page_id = id

    def addZone(self, zone):
        self.zones.append(zone)
        self.text += zone.getFullText()

    def getFullText(self):
        """ Return the full plaintext from the page """
        # full_text = ''
        # for zone in self.zones:
        #     full_text += zone.getFullText()
        # return full_text
        return self.text


class Zone:
    """
    Representation of a parsed zone, containing multiple lines
    Zones have labels and bounding boxes
    """

    def __init__(self, zone_id):
        self.zone_id = zone_id
        self.label = ""
        self.top_left = (None, None)
        self.bottom_right = (None, None)
        self.lines = []
        self.text = ""

    def setID(self, id):
        self.zone_id = id

    def setLabel(self, label):
        self.label = label

    def setTopLeft(self, x, y):
        self.top_left = (x, y)

    def setBottomRight(self, x, y):
        self.bottom_right = (x, y)

    def addLine(self, line):
        self.lines.append(line)
        self.text += line.getFullText() + "\n"

    def getFullText(self):
        """ Return the full plaintext from the zone """
        # full_text = ''
        # for line in self.lines:
        #     full_text += line.getFullText()
        #     full_text += '\n'
        # return full_text
        return self.text


class Line:
    """
    Representation of a line
    Lines have labels (in our case, the same as zones) and bounding boxes
    """
    def __init__(self, line_id):
        self.line_id = line_id
        self.label = ""
        self.top_left = (None, None)
        self.bottom_right = (None, None)
        self.words = []
        self.text = ""

    def setID(self, id):
        self.line_id = id

    def addWord(self, word):
        self.words.append(word)
        self.text += word.text + " "

    def setLabel(self, label):
        self.label = label

    def setTopLeft(self, x, y):
        self.top_left = (x, y)

    def setBottomRight(self, x, y):
        self.bottom_right = (x, y)

    def getFullText(self):
        """ Return the full plaintext from the line """
        # full_text = ''
        # for word in self.words:
        #     full_text += word.text + " "
        # return full_text
        return self.text

class Word:
    """
        Representation of a word
        Words have labels (in our case, the same as enclosing lines) and bounding boxes
    """

    def __init__(self, word_id):
        self.word_id = word_id
        self.text = ""
        self.label = ""
        self.top_left = (None, None)
        self.bottom_right = (None, None)

    def setID(self, id):
        self.word_id = id

    def setLabel(self, label):
        self.label = label


    def setTopLeft(self, x, y):
        self.top_left = (x, y)


    def setBottomRight(self, x, y):
        self.bottom_right = (x, y)

    def setText(self, text):
        self.text = text