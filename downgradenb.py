# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:51:00 2015

@author: Peter Wittek
"""
import io
import os
import sys
import IPython.nbformat as nbformat

fname = sys.argv[1]
if len(sys.argv) == 3:
    target_version = int(sys.argv[2])
else:
    target_version = 3

base, ext = os.path.splitext(fname)
newname = base+'.v'+str(target_version)+ext
print "downgrading %s -> %s" % (fname, newname)
with io.open(fname, 'r', encoding='utf8') as f:
    nb = nbformat.read(f, target_version)
with open(newname, 'w') as f:
    nbformat.write(nb, f, target_version)
