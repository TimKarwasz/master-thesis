#!/usr/bin/env python
# This code was taken from https://github.com/neubig/util-scripts/tree/master and updated to modern python

'''
A program to calculate syntactic complexity of parse trees. (Relies on NLTK)
This is an implementation of some of the methods in:

Syntactic complexity measures for detecting Mild Cognitive Impairment
Brian Roark, Margaret Mitchell and Kristy Hollingshead
Proc BioNLP 2007.
'''

import sys
import os
from nltk.tree import *


def calc_words(t):
    if type(t) == str:
        return 1
    else:
        val = 0
        for child in t:
            val += calc_words(child)
        return val

def calc_nodes(t):
    if type(t) == str:
        return 0
    else:
        val = 0
        for child in t:
            val += calc_nodes(child)+1
        return val

def calc_yngve(t, par):
    if type(t) == str:
        return par
    else:
        val = 0
        for i, child in enumerate(reversed(t)):
            val += calc_yngve(child, par+i)
        return val

def is_sent(val):
    return len(val) > 0 and val[0] == "S"

def calc_frazier(t, par, par_lab):
    if type(t) == str:
        return par-1
    else:
        val = 0
        for i, child in enumerate(t):
            # For all but the leftmost child, zero
            score = 0
            if i == 0:
                my_lab = t.label()
                # If it's a sentence, and not duplicated, add 1.5
                if is_sent(my_lab):
                    score = (0 if is_sent(par_lab) else par+1.5)
                # Otherwise, unless it's a root node, add one
                elif my_lab != "" and my_lab != "ROOT" and my_lab != "TOP":
                    score = par + 1
            val += calc_frazier(child, score, my_lab)
        return val

def calc_scores(t):
    sents = 0
    words_tot = 0
    yngve_tot = 0
    frazier_tot = 0
    nodes_tot = 0

    #t = Tree.fromstring('(S (NP I) (VP (V enjoyed) (NP my cookie)))')
    words = calc_words(t)
    words_tot += words
    sents += 1
    yngve = calc_yngve(t, 0)
    yngve_avg = float(yngve)/words
    yngve_tot += yngve_avg
    nodes = calc_nodes(t)
    nodes_avg = float(nodes)/words
    nodes_tot += nodes_avg
    frazier = calc_frazier(t, 0, "")
    frazier_avg = float(frazier)/words
    frazier_tot += frazier_avg
    yngve_avg = float(yngve_tot)/sents
    frazier_avg = float(frazier_tot)/sents
    nodes_avg = float(nodes_tot)/sents
    words_avg = float(words_tot)/sents
    

    #print(f"Total frazier = {frazier_tot}, avg frazier = {frazier_avg}, total yngve = {yngve_tot} ,  avg yngve = {yngve_avg} ")
    return frazier_tot,yngve_tot

