# -*- coding: utf-8 -*-


import os
import codecs
import string

class StoriesCollection:
    def __init__(self, data_dir):
        self.dir = data_dir + "\\stories"
        self.ann_sign = "@highlight"
        self.ids = os.listdir(self.dir)
        self.cur_id = 0
        self.paragraphs = []
        self.header = ""
        self.cycle = False
        
    def process_ann(self, ann):
        return ann.replace(self.ann_sign, " ")
    
    def increase_cur_id(self):
        self.cur_id += 1
        if self.cur_id >= len(self.ids):
            self.cur_id = 0
            self.cycle = True

    def divide_into_parts(self, text):
        story_end = text.find(self.ann_sign)
        story = text[:story_end]
        if (story.strip('\n') == ""):
            self.increase_cur_id()
            return self.get_story(self.ids[self.cur_id])
        lines = story.split('\n')
        while (self.header == ""):
            self.header = lines[0]
            lines = lines[1:]
        ann = self.process_ann(text[story_end:])
        return (lines, ann)        

    def get_story(self, id_story):
        #ileObj = codecs.open( "someFilePath", "r", "utf_8_sig" )
        #text = fileObj.read() # или читайте по строке
        #fileObj.close()
        
        with codecs.open(self.dir + "\\" + id_story, "r", "utf_8_sig") as file:
            text = file.read()
            return self.divide_into_parts(text)

    def get_list_of_ids(self):
        return self.ids
    
    def get_paragraphs(self, id_story):
        """
        segment the raw text into paragraphs
        """
        (lines, ann) = self.get_story(id_story)
        
        graf = []
        for line in lines:
            line = line.strip()
    
            if len(line) < 1:
                if len(graf) > 0:
                    yield "\n".join(graf)
                    graf = []
            else:
                graf.append(line)
    
        if len(graf) > 0:
            yield "\n".join(graf)
            
    def create_corpus(self, id_story):
        self.paragraphs = list(self.get_paragraphs(id_story))
        corpus = []
        translate_table = str.maketrans(dict.fromkeys(string.punctuation))
        for paragraph in self.paragraphs:
            paragraph = paragraph.translate(translate_table).lower()
            corpus.append(paragraph.split(' '))
        self.header = self.header.translate(translate_table).lower()
        return corpus
    
    def get_next_corpus(self):
        corpus = self.create_corpus(self.ids[self.cur_id])
        self.increase_cur_id()
        return corpus
    
    def get_next_corpus_textrank(self):
        with codecs.open(self.dir + "\\" + self.ids[self.cur_id], "r", "utf_8_sig") as file:
            text = file.read()
            story_end = text.find(self.ann_sign)
            story = text[:story_end]
            if (story.strip('\n') == ""):
                self.increase_cur_id()
                return self.get_next_corpus_textrank()     
            self.increase_cur_id()
            return story
    
    def get_last_paragraphs(self, paragraphs_list):
        required_paragraphs = []
        for cur_paragraph in sorted(paragraphs_list):
            required_paragraphs.append(self.paragraphs[cur_paragraph])
        return required_paragraphs
    
    def get_cur_doc_name(self):
        return self.ids[self.cur_id - 1]
    
    def get_header(self):
        return self.header
    
    def get_n_files(self):
        return len(self.ids)
    
    def get_cur_id(self):
        return self.cur_id - 1
    
    def was_cycle(self):
        if self.cycle:
            self.cycle = False
            return True
        return self.cycle