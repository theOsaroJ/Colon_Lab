#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math

data=pd.read_csv('Finaldata.csv', delimiter=',')
bin=data.iloc[:,0].values
dd=data.iloc[:,1].values

#plotting
plt.plot(bin,dd)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Pore Size[A]',fontsize=17)
plt.ylabel('Derivative [Arb. Units]',fontsize=17)
plt.savefig('Histogram.png')
plt.savefig('Histogramm.pdf')
