import numpy as np
import sys
sys.path.append('/Users/sfregoso/Documents/old_mac/MacPro_2012thru2015/Santos/Work/FelipeStuff/FALCO/git_repos/public/falco-python/')
import falco

class EmptyObj(object):
    pass

def AddToObject(mp):
    mp.felipe = EmptyObj()
    mp.felipe.age = 39

def AddMoreVariables(mp):
    mp.felipe.eye = EmptyObj()
    mp.felipe.eye.color = 'brown'
    mp.felipe.eye.stigmatism = False
    mp.felipe.hair = EmptyObj()
    mp.felipe.hair.color = 'black'
    mp.felipe.hair.length = 'short'
    mp.felipe.hair.dyed = False
    
mp = falco.config.ModelParameters()
AddToObject(mp)
AddMoreVariables(mp)

print(mp.felipe.age)
print(mp.felipe.eye.color)
print(mp.felipe.eye.stigmatism)
print(mp.felipe.hair.length)
print(mp.felipe.hair.color)
print(mp.felipe.hair.dyed)
