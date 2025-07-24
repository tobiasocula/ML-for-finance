
import sys
import os

dir = '../data/stocks_1d'

for filename in os.listdir(dir):
    new = filename.replace("_", '--')
    os.rename(dir + '/' + filename, dir + '/' + new)