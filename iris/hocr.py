# -*- coding: utf-8 -*-
import os
import re
from lxml import etree
from kitchen.text.converters import to_unicode, to_bytes
from PIL import Image, ImageDraw

# Useful xpath queries for selecting items with bboxes from hocr.
ALL_BBOXES = u"//*[@title]"
PAGES = u"//*[@class='ocr_page' and @title]"
LINES = u"//*[@class='ocr_line' and @title]"
WORDS = u"//*[@class='ocrx_word' and @title]"


class HocrContext(object):
    """
    A context manager for working with parsed hocr.
    """
    def __init__(self, hocrfilepath):
        super(HocrContext, self).__init__()
        self.hocrfilepath = hocrfilepath

    def __enter__(self):
        abspath = os.path.abspath(os.path.expanduser(self.hocrfilepath))
        print 'absp: ' + abspath
        with open(abspath) as hocrfile:
            self.parsedhocr = etree.parse(hocrfile)
            return self.parsedhocr


    def __exit__(self, type, value, traceback):
        del self.parsedhocr
        return False    # No exception suppression.
        # self.cr.restore()

        


def extract_hocr_tokens(hocr_file):
    """
    Extracts all the nonempty words in an hOCR file and returns them
    as a list.
    """
    words = []
    context = etree.iterparse(hocr_file, events=('end',), tag='span', html=True)
    for event, element in context:
        # Strip extraneous newlines generated by the ocr_line span tags.
        if element.text is not None:
            word = to_unicode(element.text.rstrip())
        if len(word) > 0:
            words.append(word)
        element.clear()
        while element.getprevious() is not None:
            del element.getparent()[0]
    del context
    return words

def extract_bboxes(hocr_file, xpaths=[ALL_BBOXES]):
    """
    Extracts a list of bboxes as 4-tuples, in the same order that they
    appear in the hocr file. BBoxes are only extracted from those
    elements matching the specified xpath bboxes.
    """
    context = etree.parse(hocr_file)
    bboxpattern = r'.*(bbox{1} [0-9]+ [0-9]+ [0-9]+ [0-9]+)'
    results = {}
    for xpath in xpaths:
        bboxes = []
        for e in context.xpath(xpath):
            match = re.match(bboxpattern, e.attrib[u'title'])
            bbox = tuple(map(int, match.groups()[0][5:].split(u' ')))
            bboxes.append(bbox)
        results[xpath] = bboxes

    return results

def drawbboxes(bboxes, pil_img, color='blue'):
    """
    Draw all bboxes in the specified color. Returnss a 
    """
    draw = ImageDraw.Draw(pil_img)
    for bbox in bboxes:
        draw.rectangle(((bbox[0], bbox[1]),(bbox[2], bbox[3])), outline=color)
    del draw
    return pil_img

def previewbboxs(imgfile, hocrfile, color='blue'):
    """
    Display a preview of the specified image with the bboxes from the
    hocr file drawn on it.
    """
    opened = Image.open(imgfile)
    drawbboxes(extract_bboxes(hocrfile)[ALL_BBOXES], opened, color)
    opened.show()

def markbboxes(imgfile, hocrfile, tag_color_dict):
    """
    Draw all the bboxes of the specified hocr class with the specified
    colors. Returns a PIL image file. Tag_color_dict is a dictionary of the
    form {'hocrclass':'color'}.
    """
    # bboxesperclass = extract_bboxes_by_classes(hocrfile, tag_color_dict.keys())
    bboxesperclass = extract_bboxes(hocrfile, tag_color_dict.keys())
    pil_img = Image.open(imgfile)
    for hocr_class, bboxlist in bboxesperclass.iteritems():
        drawbboxes(bboxlist, pil_img, tag_color_dict[hocr_class])

    pil_img.show()
    return pil_img

# def detect_word_lang(hocrfile, uni_blocks, threshold=1.0):
    
    

if __name__ == '__main__':
    pass