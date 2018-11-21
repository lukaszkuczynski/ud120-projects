# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 06:42:45 2018
"""

from math import log



entropy = -2.0/3.0 * log(2.0/3.0, 2) - 1.0/3.0 * log(1.0/3.0, 2)
#entropy = 2 / 3.0
print("entropy = %f" % entropy)