""" Jargon: Simple language utilities. """

__version__ = '0.1dev'

import nltk
import numpy
import warnings
from re import match

stemmer = nltk.stem.snowball.EnglishStemmer()

def voice(sentence):
    ##-- determine if a sentence is (probably) in "active" or "passive" voice
    ##-- return 1 if active, 0 if passive, -1 if indeterminate (rare)
    
    if len(nltk.sent_tokenize(sentence)) > 1:
        warnings.warn("Subroutine voice() only accepts single setences.", UserWarning)
        return -1
    
    tags0  = numpy.asarray( nltk.pos_tag(nltk.word_tokenize(sentence)) )
    tags = tags0[ numpy.where( -numpy.in1d( tags0[:,1], ['RB', 'RBR', 'RBS', 'TO'] ) ) ] #<-- remove adverbs, 'TO'

    if len(tags) < 2: # <-- too short to really know.
        return -1
    
    to_be = ['be','am','is','are','was','were','been','has','have','had','do','did','does','can','could','shall','should','will','would','may','might','must']

    WH = [ 'WDT', 'WP', 'WP$', 'WRB' ]
    VB = ['VBG', 'VBD', 'VBN', 'VBP', 'VBZ', 'VB']
    VB_nogerund = ['VBD', 'VBN', 'VBP', 'VBZ']
    
    logic0 =  numpy.in1d(tags[:-1,1],['IN'])*numpy.in1d(tags[1:,1],WH) # <-- passive if true
    if numpy.any(logic0):
        return 0

    logic1 = numpy.in1d(tags[:-2,0],to_be)*numpy.in1d(tags[1:-1,1],VB_nogerund)*numpy.in1d(tags[2:,1],VB) #<-- chain of three verbs, active if true and previous not
    if numpy.any(logic1):
        return 1 
    
    if numpy.any(numpy.in1d(tags[:,0],to_be))*numpy.any(numpy.in1d(tags[:,1],['VBN'])): ## <-- 'to be' + past participle verb
        return 0 

    ##-- if no clauses have tripped thus far, it's probably active voice:
    return 1 

def active_fraction(text, return_tags=False):
    ##-- returns fraction of active / passive sentences
    ##-- if return_tags == True,
    ##--    return the sentences and their tags (active or passive)
    
    sent = nltk.sent_tokenize(text)

    tags,full_tags = [],[]
    for sentence in sent:
        tags.append(voice(sentence))
        full_tags.append([sentence, tags[-1]])

    tags = numpy.asarray( tags )
    if numpy.any( tags != -1):
        frac = numpy.mean(tags[ numpy.where(tags != -1) ])
    else:
        ##-- no confident classifications, return -1
        frac = -1
        
    if return_tags:
        return (frac, full_tags,)
    else:
        return frac

def wreduce(word):
    ##-- lower-case, strip periods, commas, apostrophes, etc.
	return ''.join([x for x in word.lower() if match(r'\w', x)])

def unique_word_roots( text, full_output=False ):
    ##-- count the total number of unique word roots in supplied text
    
    words0 = text.split()
    words = []

    ##-- split hyphenated words
    for word in words0:
        if word.count('-')>0:
            words2 = word.split('-')
            for word2 in words2:
                words.append( word2 )
        else:
            words.append( word )
    
    seen_words = set()
    unique = []
    for word0 in words:
        word =  stemmer.stem( wreduce(word0) ) # <-- only keep word root
        if word not in seen_words:
            seen_words.add( word )

    return len(seen_words), len(words)

    
