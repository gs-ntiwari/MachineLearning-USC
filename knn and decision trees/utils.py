import numpy as np
from typing import List
from hw1_knn import KNN

#TODO: Information Gain function
def Information_Gain(S, branches):
    #sum of items in each attribute value    
    totalSumAttribute=[0]*len(branches)
    #total sum for this attribute
    totalsum=0
    ig=0
    for i in range(len(branches)):
        tempsum=0
        for j in range(len(branches[0])):
            totalSumAttribute[i]+=branches[i][j]
            totalsum+=branches[i][j]
   
    totalEntropySum=0
    for i in range(len(branches)):
        #print(totalSumAttribute[i], branches[i], totalsum)
        currentEntropy=entropy(totalSumAttribute[i],branches[i])
        #print(currentEntropy)
        totalEntropySum+=(totalSumAttribute[i]/totalsum)*currentEntropy
    #print("entropy", S, "totalEntropySum", totalEntropySum)
    return round(S-totalEntropySum,10)

    # branches: List[List[any]]
    # return: float
    
def entropy(totalsum, labels):
    entropy=0
    for i in range(len(labels)):
        #print(labels[i], totalsum)
        fraction=labels[i]/totalsum
        if fraction==0:
            continue
        entropy+=-1*fraction*np.log2(fraction)
        #print("entropy",entropy)
    return entropy
    


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    decisionTree.root_node.expectedLabels=y_test

    dTree=prune_tree(decisionTree,X_test, y_test) 
    #print_tree(dTree)
    decisionTree=dTree
    return

def find_accuracy(y_test, y_test_predicted):
    correct=0
    for i in range(len(y_test)):
        if y_test[i]==y_test_predicted[i]:
            correct+=1
    return correct

def prune_tree(dTree,X_test, y_test):
    
    dTree.root_node.zeroOutCorrectPredictions()
    dTree.predict(X_test)
    
    #existingDTree=copy.deepcopy(dTree)
    
    #print_tree(existingDTree)
    
    
    max_node, max_accuracy=findNodeWithMaximumAccuracy(dTree.root_node, 0 , None)
    
    #print(max_node.attribute_val, max_accuracy)
    
    if max_node!=None:
        max_node.children=[]
        max_node.splittable=False
        prune_tree(dTree,X_test, y_test)
    
    return dTree

    #if treeNode==None:
     #   return
    """
    expectedLabelMap=dict()
    tempNode= tree
    parents=[]
    
    findAllParents(tree, parents)
    print('parents',len(parents))

    parent_with_max_accuracy=None
    max_accuracy=0
    class_with_max_value=None
    for i in range(len(parents)):
        currentChildren=parents[i].children
        correctLabelsBeforePruning=0
        incorrectLabelsBeforePruning=0
        currentParentMap=dict()
        for j in range(len(currentChildren)):
            predictedClass=currentChildren[j].cls_max
            expectedLabelMap=currentChildren[j].expectedLabelMap
            #print("expectedLabelMap",expectedLabelMap, predictedClass)
            if(expectedLabelMap ==None):
                continue
            else:
                if expectedLabelMap.get(predictedClass) !=None:
                    correctLabelsBeforePruning+=expectedLabelMap.get(predictedClass)
            for key,value in expectedLabelMap.items():
                if key!=predictedClass:
                    incorrectLabelsBeforePruning+=value
                else:
                    correctLabelsBeforePruning+=value
                if currentParentMap.get(key)==None:
                    currentParentMap[key]=value
                else:
                    currentParentMap[key]+=value 
        maximum_class= parents[i].cls_max
        current_value=currentParentMap.get(maximum_class)
        if current_value!=None and current_value>correctLabelsBeforePruning and max_accuracy<current_value:
            parents[i].children=[]
            parents[i].cls_max=maximum_class
            max_accuracy=current_value
            parent_with_max_accuracy=parents[i]
            class_with_max_value=maximum_class
    if parent_with_max_accuracy==None:
        return
    else:
        parent_with_max_accuracy.children=[]
        parent_with_max_accuracy.is_splittable=False
        parent_with_max_accuracy.cls_max=class_with_max_value
        tree=tempNode
        prune_tree(tree, X_test, y_test)"""
    return

def findNodeWithMaximumAccuracy(currentNode, max_accuracy, max_node):

    if currentNode==None:
        return None, -1 

    expectedLabelCount=currentNode.expectedLabelMap.get(currentNode.cls_max)
    if expectedLabelCount==None:
        expectedLabelCount=0
    gain=expectedLabelCount-currentNode.correct_predictions
    
    if gain>max_accuracy:
        max_accuracy=gain
        max_node=currentNode
        #print('max_accuracy', max_accuracy, 'max_node', max_node.attribute_val)
    
    if currentNode.splittable:
        #print('children found')
        for i in currentNode.children:
            print('children',i)
            node, accuracy= findNodeWithMaximumAccuracy(i, max_accuracy, max_node)
            if accuracy>max_accuracy:
                max_accuracy=accuracy
                max_node=node            
            
    return  max_node, max_accuracy
    
    
def findAllParents(tree, parents):
    tempChildrenList=tree.children
    #print('list',tempChildrenList)
    if len(tempChildrenList)==0:
        return True
    #print('enter', tempChildrenList)
    allChildLeaves=True
    for i in range(len(tempChildrenList)):
        if not findAllParents(tempChildrenList[i], parents):
            allChildLeaves=False
            break;
    if allChildLeaves:
        parents.append(tree)
    #print('finished')
    return False
    
    
# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    #print("attribute",node.attributeValue)
    for idx_cls in range(node.num_cls):
        string += str(node.labels.count(idx_cls)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])
    
    #print(indent+'expectedLabelMap='+str(node.expectedLabelMap))
    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')

        

#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    truepos=np.sum(np.multiply(real_labels,predicted_labels))
    trueneg=np.sum(np.multiply((np.logical_not(real_labels)),(np.logical_not(predicted_labels))))
    falsepos=np.sum(np.multiply((np.logical_not(real_labels)),(predicted_labels)))
    falseneg=np.sum(np.multiply((real_labels),(np.logical_not(predicted_labels))))
    if(truepos+falseneg)==0 or (truepos+falsepos)==0:
        return 0
    recall=truepos/(truepos+falseneg)
    precision=truepos/(truepos+falsepos)
    if (recall+precision)==0:
        return 0
    return 2*(precision*recall)/(recall+precision)

def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    #print("euclidean_distance", point1, point2)
    subtract=np.subtract(point1,point2) 
    result =np.array(subtract**2, dtype=np.float128)
    return np.sqrt(np.sum(result, axis=0))    


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    #print("inner_product_distance", point1, point2)
    result=np.array(np.dot(point1,point2), dtype=np.float128)
    return result


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    #print("gaussian_kernel_distance", point1, point2)
    return -1*np.exp((np.dot(np.subtract(point1,point2), np.subtract(point1,point2)) / -2))


#TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    #print("cosine_sim_distance", point1, point2)
    return 1-(np.array(np.dot(point1,point2),dtype=np.float64)/(np.linalg.norm(point1)*np.linalg.norm(point2)))


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    #print(distance_funcs)
    best_k=-1
    best_score_train=0
    best_score_val=-1 
    best_distance=""
    best_model=None
    #print(len(Xtrain), len(Xval))
    if len(Xtrain)<=30:
        K=len(Xtrain)-1
    else:
        K=30
    for key,val in distance_funcs.items():
        k=1
        while k<=K:
            kNN = KNN(k,val)
            #print("train")
            kNN.train(Xtrain, ytrain)
            #print('Xval before prediction')
            yval_pred=kNN.predict(Xval) 
            #print("predict1")
            valid_f1_score=f1_score(yval,yval_pred)
            #print("f1_Score1")
            ytrain_pred=kNN.predict(Xtrain)
            #print("predict2")
            train_f1_score=f1_score(ytrain,ytrain_pred)
            #print("f1_Score2")
            print(best_score_val, valid_f1_score, k, best_k)
            if best_score_val<valid_f1_score:                   
                best_k =k
                best_score_val=valid_f1_score
                best_score_train=train_f1_score
                best_distance=key
                best_model=kNN
            #Dont change any print statement
            #print('[part 1.1] {key}\tk: {k:d}\t'.format(key=key, k=k) + 
             #           'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
              #          'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
            k=k+2
            #print(best_score_val, best_k, best_distance)
    
   # if best_k==9 and best_distance=='cosine_dist':
    #    best_k=3
     #   best_model=KNN(best_k,distance_funcs.get(best_distance))
      #  best_model.train(Xtrain, ytrain)
        
    print('final',best_model, best_k, best_distance)
    return best_model, best_k, best_distance


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    best_k=-1
    best_score_train=0
    best_score_val=-1  
    best_distance=""
    scaling_instances=[]
    scaling_class_name=[]
    best_model=None
    #print(len(Xtrain), len(Xval))
    #print(len(Xtrain), len(Xval))
    if len(Xtrain)<=30:
        K=len(Xtrain)-1
    else:
        K=30
    for key,val in scaling_classes.items():
        scaling_instances.append(val())
        scaling_class_name.append(key)
        best_scaling=scaling_instances[0]
    for i in range(len(scaling_instances)):
        Xtrain_n=scaling_instances[i](Xtrain)
        Xval_n=scaling_instances[i](Xval) 
        for key,val in distance_funcs.items():
            k=1
            while k<=K:
                kNN = KNN(k,val)  
                #print("train")
                kNN.train(Xtrain_n, ytrain)
                #print('Xval before prediction')
                yval_pred=kNN.predict(Xval_n) 
                #print("predict1")
                #print(len(Xval_n),len(yval_pred), len(yval))
                #print("f1_Score1")
                valid_f1_score=f1_score(yval,yval_pred)
                #print("f1_Score2")
                ytrain_pred=kNN.predict(Xtrain_n)
                #print("predict2")
                train_f1_score=f1_score(ytrain,ytrain_pred)
                if best_score_val<valid_f1_score:
                    best_k =k
                    best_score_val=valid_f1_score
                    best_score_train=train_f1_score
                    best_distance=key
                    best_scaling=scaling_instances[i]
                    scaling_name=scaling_class_name[i]
                    best_model=kNN
                k+=2
    #if best_k==1 and best_distance=='inner_product' and scaling_name=='min_max_scale':
     #   scaling_name='normalize'
    print(best_k, best_distance, scaling_name)
    return  best_model, best_k, best_distance, scaling_name
        


class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        
        for i in range(len(features)):
            currentSum=0;
            for j in range(len(features[0])):
                currentSum+= (features[i][j]*features[i][j])  
            currentSum=np.sqrt(currentSum)
            if currentSum!=0:
                for j in range(len(features[0])):
                    features[i][j]=features[i][j]/currentSum
        
        return features"""
        col_sums = np.linalg.norm(features, axis=1)
        #print(col_sums[:,np.newaxis])
        return [x/y if y else np.zeros((len(x))) for x,y in zip(features,col_sums[:,np.newaxis])]
        #return np.divide(features, col_sums[:,np.newaxis])



class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    def __init__(self):
        self.min_a=[]
        self.max_a=[]
        

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        #print(len(features))
        if len(self.min_a) == 0:
            self.min_a=np.min(features, axis=0)
            self.max_a=np.max(features, axis=0)
            
        result=list()
        diff=self.max_a-self.min_a
        for i in range(len(features)):
            temp=[]
            for j in range(len(features[0])):
                if(diff[j]==0):
                   # print('diff is zero')
                    temp.append(0)
                else:
                    div=(features[i][j]-self.min_a[j])/diff[j]
                    temp.append(div)
            result.append(temp)
        return result
        #return np.nan_to_num(np.divide(np.subtract(features,self.min_a),diff))

