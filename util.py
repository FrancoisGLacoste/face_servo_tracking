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

