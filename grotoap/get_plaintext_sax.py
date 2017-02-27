import argparse
import codecs
import os
from xml.sax import make_parser, ContentHandler
from xml.sax.handler import feature_namespaces, feature_external_ges

# SAX content handler to assemble words and lines from the characters, print them to the specified file
class ConstructLine(ContentHandler):
    def __init__(self, output_filename):
        # self.output_file = codecs.open(output_filename, 'w', 'utf-8')
        self.output_file = output_filename
        self.lineText = ''
        self.wordText = ''

    def startElement(self, name, attrs):
        # Start new line and add old line to buffer
        if name == 'Line':
            self.lineText += self.wordText
            if self.lineText != '':
                # todo: changed this to just write one doc per line
                self.output_file.write(self.lineText + " ") #+ '\n')
            self.lineText = ''
            self.wordText = ''

        # Start new word and add old word to line
        elif name == 'Word':
            if self.wordText != '':
                self.lineText += self.wordText + ' '
            self.wordText = ''

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

def main():
    parser = make_parser()
    parser.setFeature(feature_namespaces, False)
    parser.setFeature(feature_external_ges, False)

    arg_parser = argparse.ArgumentParser(description='Extract original plaintext from TrueViz documents')
    arg_parser.add_argument('trueviz_directory_path')
    arg_parser.add_argument('target_directory_path')
    arg_parser.add_argument('-f', action='store_true', default=False,
                            help='write all the text to one file instead of separate files')
    args = arg_parser.parse_args()

    trueviz_dir_path = args.trueviz_directory_path
    target_dir_path = args.target_directory_path

    # create a directory to hold the plaintext files
    if not args.f and not os.path.exists(target_dir_path):
        os.makedirs(target_dir_path)
    # or if writing all the text to one file, clear it first
    else:
        open('grotoap-full.txt', 'w').close()

    if not os.path.exists(trueviz_dir_path):
        print("Incorrect path to TrueViz directory: %s" % trueviz_dir_path)
        return
    else:
        num_files = 0
        # walk over GROTOAP dataset directory
        for root, subdirs, files in os.walk(trueviz_dir_path):
            for tv_file in files:
                if '.cxml' in tv_file:
                    num_files += 1
                    text_file_path = target_dir_path + os.sep + tv_file
                    src_file_path = root + os.sep + tv_file
                    if not args.f:
                        print("Writing file %s" % text_file_path)
                        with codecs.open(text_file_path, 'w', 'utf-8') as f:
                            dh = ConstructLine(f)
                            parser.setContentHandler(dh)
                            parser.parse(src_file_path)
                    else:
                        with codecs.open('grotoap-full.txt', 'a', 'utf-8') as f:
                            dh = ConstructLine(f)
                            parser.setContentHandler(dh)
                            parser.parse(src_file_path)

        if args.f:
            output_type = 'file'
            path = 'grotoap-full.txt'
        else:
            output_type = 'directory'
            path = target_dir_path
        print("Created %s containing %d parsed docs: %s" % (output_type, num_files, path))

if __name__ == '__main__':
    main()