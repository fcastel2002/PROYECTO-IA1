import numpy
import matplotlib.pyplot as plt
from Archivos import *

class KmeansClustering:
    def __init__(self, k = 4):
        self.k = k
        self.centroids = None
    
    def fit(self, X, iteraciones_max = 200):
        
        self.centroids = np.random.uniform
        