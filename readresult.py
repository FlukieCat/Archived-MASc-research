# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 22:37:10 2019

@author: Yue
"""

import dill

with open('results.dill', 'rb') as input_file:
    dill.load_session(input_file)