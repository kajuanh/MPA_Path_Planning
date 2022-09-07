import numpy as np
from numpy import random
from numpy.random.mtrand import beta
import initialization as ini
import Get_Functions_details as GFd
import math
import levy
import pandas as pd
ludf = GFd.switcher(1)
fobj = ludf['fobj']
data = pd.read_csv(
    "D:\downloads\Marine-Predators-Algorithm-master\Marine-Predators-Algorithm-master\dtnet.csv", names=['x1', 'x2', 'x3', 'x4'])
Prey = np.array(data).reshape(4, 50)
fis = ludf['fobj']