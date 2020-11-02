import matplotlib.pyplot as plt
import numpy as np

blob_width = 300
blob_height = 300
blob_x = -200

filename = str(blob_width) + '_' + str(blob_height) + '_' + str(blob_x) + '_' + 'vertical.txt'

blob_ys, peaks = np.loadtxt(filename, unpack = True, usecols = (0, 1))


