import sys, os
import pdb
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
pos='t-scend'
path = os.getcwd()
SRC_PATH = path.split('X_src')[0]+"X_src/"
CURRENT_WP=path.split('X_src')[0]
if pos=='t-scend':
    EXP_PATH = CURRENT_WP
else:
    raise ValueError("Please specify the position of the project directory")
