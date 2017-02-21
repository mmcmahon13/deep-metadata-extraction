# Classes to represent the document structure, as parsed out of the TrueViz XML format

class Document:
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.pages = []

    def getFullText(self):
        full_text = ''
        for page in self.pages:
            full_text += page.getFullText()
        return full_text

    def toString(self):
        s = ''
        for page in self.pages:
            s += "Page %d\n" % page.page_id
            for zone in page.zones:
                s += "\tZone %d\n" % zone.zone_id
                s += "\tZone class: %s\n" % zone.label
                for line in zone.lines:
                    s += "\t\tLine %d\n" % line.line_id
                    s += "\t\t%s\n" % line.text
        return s

class Page:
    def __init__(self, page_id):
        self.page_id = page_id
        self.zones = []

    def getFullText(self):
        full_text = ''
        for zone in self.zones:
            full_text += zone.getFullText()
        return full_text

class Zone:
    def __init__(self, zone_id, label, top_left, bottom_right):
        self.zone_id = zone_id
        self.label = label
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.lines = []

    def getFullText(self):
        full_text = ''
        for line in self.lines:
            full_text += line.text
            full_text += '\n'
        return full_text

class Line:
    def __init__(self, line_id, label, top_left, bottom_right, text):
        self.line_id = line_id
        self.label = label
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.text = text
