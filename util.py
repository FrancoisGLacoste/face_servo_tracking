# -*- encoding: utf-8 -*-

import numpy as np

def argmax_dict(d):
    """ 
    Arg: 
        d:  dictionary {key: value, ...} where values are numbers. 
    returns
        The key corresponding to the max value 
    """
    id = np.argmax(d.values())
    return list(d.keys())[id]


def argmax_tupls(l, ind = 1):
    """
    Arg:
        l:  list of tuples: [tupl0, tupl1, ....]
        where tupl0 = (tupl00, tupl01 )
        
    returns
        tupl  such that tupl[ind] is max among all tuples in the list    
    """
    index = np.argmax([tupl[ind] for tupl in l ])
    return l[index], index