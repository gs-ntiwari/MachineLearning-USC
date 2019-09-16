from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        self.train_f=features
        self.train_l=labels
        return

    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        try:
            pred_list=list()
            #print("predict entered")
            for i in range(len(features)):
                count_for_labels= dict()
                k_neighbours=self.get_k_neighbors(features[i])
                pred_list.append(max(k_neighbours,key=k_neighbours.count))
                #pred_list.append(maxcount_label)  
                #print("predict before returning ")
        except:
            print("knn predict exception")
        return  pred_list

        
    
    #TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        try:
            distances =[]
            #print("enter k neighbour")
            for i in range(len(self.train_f)):
                d=self.distance_function(self.train_f[i], point)
                #print(d)
                distances.append(d)
            #print("distances calculate", len(distances))
            #if self.distance_function.__name__ in ["inner_product_distance"]:
                #print("distances calculate with -1 started")
               # distances=np.multiply(-1,distances)
                #print("distances calculate with -1")

            all_neighbors=np.argsort(distances)
            #print("argsort")
            all_neighbors=all_neighbors[0:self.k]
            #print("argsort k")
            returnList=[]
            for i in all_neighbors:
                returnList.append(self.train_l[i])
            #print("return k neighbours")
        except:
            print("get_k_neighbors exception")
        return returnList


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
