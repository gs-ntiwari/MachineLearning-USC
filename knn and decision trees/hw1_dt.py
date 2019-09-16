import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        #print(features, labels)
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        #if self.root_node.splittable:
            #self.root_node.split()
        #Util.print_tree(self.root_node)
        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            #print(self.root_node.expectedLabels)
            if len(self.root_node.expectedLabels) !=0:
                self.root_node.currentExpectedLabel=self.root_node.expectedLabels[idx]
            #print(self.root_node.currentExpectedLabel)
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        self.children_with_attributes=None
        self.dim_split = None
        self.attribute_val=None

        #attributes for pruning
        self.expectedLabels=[]
        self.expectedLabelMap=dict()
        self.trainingLabelsCountMap=dict()
        self.currentExpectedLabel=None
        self.correct_predictions=0
        self.parentNode=None

        # find the most common labels in current node
        count_max = 0
        labels_with_count=np.unique(labels, return_counts=True)
        for i in range(len(labels_with_count[0])):
            if labels_with_count[1][i] > count_max:
                count_max = labels_with_count[1][i]
                self.cls_max = labels_with_count[0][i]

        #print('treenode',self.features, num_cls,self.labels)

        # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        if len(self.features[0])==0 or len(self.features)==0:
            #print('max_class', self.cls_max)
            self.splittable = False
            return

        indexMap=self.getIndexMap(np.unique(labels))

        listOfLabelCounts=[0] * num_cls
        trainingLabelsCountMap=[0] * num_cls
        for label in labels:
            listOfLabelCounts[indexMap.get(label)]+=1
            if self.trainingLabelsCountMap.get(label)==None:
                self.trainingLabelsCountMap[label]=1
            else:
                self.trainingLabelsCountMap[label]+=1

        entropy=Util.entropy(len(labels),listOfLabelCounts)
        #print("entropy:",entropy)   

        max_ig=-1
        feature_index=None
        max_num_attributes=[]
        values=None
        for attribute in range(len(features[0])):
            num_attributes=[]
            branches=[]
            values = dict() 
            for training_point in range(len(features)):
                labelCountsFetched=values.get(features[training_point][attribute])
                if labelCountsFetched != None:
                    if labels[training_point] in labelCountsFetched:  
                        currentLabelCount=labelCountsFetched.get(labels[training_point])
                        labelCountsFetched[labels[training_point]]=currentLabelCount+1
                    else:
                        labelCountsFetched[labels[training_point]]=1
                    values[features[training_point][attribute]] =labelCountsFetched
                else:
                    labelCounts = dict()
                    labelCounts[labels[training_point]] =1    
                    values[features[training_point][attribute]] =labelCounts

            #num_attributes=[row[attribute] for row in features]
            num_attributes=np.sort(list(values.keys()))
            #print("num_attributes",num_attributes, "features", features)
            #{'a': {0: 1}, 'b': {0: 1, 1: 1}, 'c': {1: 1}}

            for key,value in values.items():
                newList=[0] * num_cls
                for k,v in value.items():
                    newList[indexMap.get(k)]=v
                branches.append(newList)
                #print(branches)
            ig=Util.Information_Gain(entropy,branches)
            print("ig:",ig, "max_ig",max_ig)
            if ig > max_ig or (ig==max_ig and len(num_attributes)>len(max_num_attributes)):
                max_ig=ig
                feature_index=attribute
                max_num_attributes=num_attributes

        self.dim_split = feature_index  # the index of the feature to be split
        #if feature_index==None:
        #print(max_ig, feature_index, labels, self.splittable)
        self.feature_uniq_split = max_num_attributes
        #print(self.feature_uniq_split)

        if max_ig>0 and self.splittable:
            #print("called")
            self.split()
        return

    #TODO: try to split current node
    def split(self):
        listOfFeatures=dict()
        listOfLabels=dict()
        if len(self.features[0])==0:
            return

        for j in range(len(self.features)):
            for i in range(len(self.feature_uniq_split)):
                featuresList=list()
                labelsList=list()
                if self.features[j][self.dim_split]==self.feature_uniq_split[i]:
                    if listOfFeatures.get(self.feature_uniq_split[i]) ==None:
                        featuresList.append(self.features[j])
                        listOfFeatures[self.feature_uniq_split[i]]=featuresList
                        labelsList.append(self.labels[j])
                        listOfLabels[self.feature_uniq_split[i]]=labelsList
                    else:
                        featuresList=listOfFeatures[self.feature_uniq_split[i]]
                        featuresList.append(self.features[j])
                        labelsList=listOfLabels[self.feature_uniq_split[i]]
                        labelsList.append(self.labels[j])
                        listOfFeatures[self.feature_uniq_split[i]]=featuresList
                        listOfLabels[self.feature_uniq_split[i]]=labelsList
                    break

        #print('split',self.feature_uniq_split, self.dim_split, self.features, self.labels, listOfFeatures,listOfLabels )            
        children_with_attributes_map=dict()

        for i in self.feature_uniq_split:
            if listOfFeatures.get(i) is not None:
                if len(self.features[0])==1:
                    newfeatures=[[]]
                else:
                    newfeatures=listOfFeatures.get(i)
                    np.delete(newfeatures, self.dim_split, 1)
                #print('check',listOfFeatures.get(i), listOfLabels[i], newfeatures, len(np.unique(listOfLabels[i])))
                currentTreeNode=TreeNode(newfeatures, listOfLabels[i], len(np.unique(listOfLabels[i])))
                currentTreeNode.attribute_val=i
                currentTreeNode.parentNode=self
                currentTreeNode.currentExpectedLabel=self.currentExpectedLabel
                self.children.append(currentTreeNode)
                children_with_attributes_map[i]= currentTreeNode
        self.children_with_attributes=children_with_attributes_map
        return
        
    def getIndexMap(self, labels):
        indexMap=dict()
        for i in range(len(labels)):
            indexMap[labels[i]]=i
        return indexMap
    
    # TODO: predict the branch or the class
    def predict(self, feature):
        #print(self.expectedLabelMap, self.currentExpectedLabel)
        self.updateExpectedLabelsMap()
        if len(self.children) ==0:    
            if self.cls_max==self.currentExpectedLabel:
                self.correct_predictions+=1
            return self.cls_max
        if self.dim_split == None:
            if self.cls_max==self.currentExpectedLabel:
                self.correct_predictions+=1
            return self.cls_max
        
        attribute_value=feature[self.dim_split]
        treeNode=self.children_with_attributes.get(attribute_value)
        
        label=None
        if treeNode != None:
            treeNode.currentExpectedLabel=self.currentExpectedLabel
            new_feature=feature
            np.delete(new_feature, self.dim_split)
            label=treeNode.predict(new_feature)
        if label !=None:
            #print(label, self.currentExpectedLabel)
            if label==self.currentExpectedLabel:
                self.correct_predictions+=1
            return label
        if self.cls_max==self.currentExpectedLabel:
                self.correct_predictions+=1
        return self.cls_max
    
    def zeroOutCorrectPredictions(self):
        if self==None:
            return
        self.correct_predictions=0
        self.expectedLabelMap=dict()
        for i in range(len(self.children)):
            self.children[i].zeroOutCorrectPredictions()
        return
    
    def updateExpectedLabelsMap(self):
        #print('updateExpectedLabelsMap',self.expectedLabelMap, self.currentExpectedLabel)
        if self.expectedLabelMap.get(self.currentExpectedLabel) ==None:
            self.expectedLabelMap[self.currentExpectedLabel]=1
        else:
            self.expectedLabelMap[self.currentExpectedLabel]+=1
        return
