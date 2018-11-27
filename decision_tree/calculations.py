# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 06:42:45 2018
"""

from math import log



entropy = -2.0/3.0 * log(2.0/3.0, 2) - 1.0/3.0 * log(1.0/3.0, 2)
#entropy = 2 / 3.0
print("entropy = %f" % entropy)


entropy_parent = 1
entropy_children = 3.0/4.0 * entropy + 1.0/4.0 * 0

information_gain = entropy_parent - entropy_children
print("information_gain = %f" % information_gain)