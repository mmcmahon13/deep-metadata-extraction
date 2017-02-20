# Classes to represent the document structure, as parsed out of the TrueViz XML format

class Document:
    def __init__(self):
        self.pages = []
        self.zones = []
        self.lines = []

    def getFullText(self):
        pass

class Page:
    def __init__(self, page_id):
        self.page_id = page_id
        self.zones = []
        self.lines = []

    def getFullText(self):
        pass

class Zone:
    def __init__(self, zone_id, label, top_left, bottom_right):
        self.zone_id = zone_id
        self.label = label
        self.rop_left = top_left
        self.bottom_right = bottom_right
        self.lines = []

    def getFullText(self):
        pass

class Line:
    def __init__(self, line_id, label, top_left, bottom_right, text):
        self.line_id = line_id
        self.label = label
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.text = text
