import numpy as np 
#import pandas as pd

class treeNode(object):
    def __init__(self, split_name, column_name, split: np.ndarray, children: np.ndarray, labels: np.ndarray, depth: int, max_depth: int=6) -> None:
        self.split_name = split_name
        self.column_name = column_name
        self.end = True if np.unique(labels).shape[0] == 1 or depth==max_depth else False
        self.depth = depth
        self.predict_value = []
        
        self.children = []
        self.children_value = np.unique(split)
        for v in self.children_value:
            self.children.append((children[split == v,:], labels[split == v]))
            value, counts = np.unique(labels[split == v], return_counts=True)
            self.predict_value.append(value[np.argmax( counts)])
            #when the tree is end due to any reasons, the predict value will be the majority class y at this branch.
        value, counts = np.unique(labels, return_counts=True)
        self.predict_value.append(value[np.argmax( counts)])
        #for x value not in self.children_value
            #observation not exist in training data. Happened in car prediction. 
            #One `person' node only have "2" "3" "4" branches, 
            #but test has "5more" 

        self.children_nodes = None

    def _set_children_node(self, children_nodes):
        self.children_nodes = children_nodes

    def predict(self, x):
        if x not in self.children_value:
            return self.predict_value[-1]
        return self.predict_value[np.argwhere(self.children_value == x).item()]

    def visit_children(self, x):
        if self.end:
            #print("This branch is end, no child.")
            return 0
        else:
            if x not in self.children_value:
                #observation not exist in training data. Happened in car prediction. 
                #One `person' node only have "2" "3" "4" branches, 
                #but test has "5more" 
                return 0
            visit = np.argwhere(self.children_value == x).item()
            #if self.children_nodes[visit].end:
            #    print("The child is the end of the branch. The predict value of {}={} is {}.".format(self.split_name, x, self.children_nodes[visit].predict_value))
            #else:
            #    print("Visit {}={}.".format(self.split_name, x))
            return visit

class DecisionTree(object):
    def __init__(self, trainx: np.ndarray, trainy: np.ndarray, column: np.ndarray, criterion: str = "entropy", max_depth: int=6, entropy_base = None) -> None:
        if criterion not in ["entropy", "gini", "me"]:
            raise ValueError("criterion should be \"entropy\", \"gini\", or \"me\"")
        if max_depth<1 or not isinstance(max_depth, int):
            raise ValueError("max_depth should be a positive integer.")
        self.criterion = criterion #"entropy", "gini", "me"
        self.max_depth = np.minimum(column.shape[0], max_depth) #should be not larger than column.shape[0]
        self.tree = None
        self.trainx = trainx
        self.trainy = trainy
        self.column = column
        self.base = entropy_base
        if self.criterion == "entropy":
            self.H = self._entropy
        elif self.criterion == "gini":
            self.H = self._gini
        elif self.criterion == "me":
            self.H = self._ME

    def _entropy(self, labels) -> np.float32:
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / np.size(labels)
        return -(probs * np.log(probs)).sum() if self.base is None else -(probs * np.log(probs)/np.log(self.base)).sum()

    def _ME(self, labels) -> np.float32:
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / np.size(labels)
        return 1-probs.max() 

    def _gini(self, labels) -> np.float32:
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / np.size(labels)
        return 1-np.square(probs).sum() 

    def _IG(self, x, labels) -> np.float32:
        IG = self.H(labels)
        values, counts = np.unique(x, return_counts=True)
        pxs = counts / np.size(x)
        for (value, px) in zip(values, pxs):
            IG -= px * self.H(labels[x == value])
        return IG
    
    def _splitter(self, datas, labels) -> None :
        n_feature = datas.shape[1]
        gain = np.zeros(n_feature)
        for i in range(n_feature):
            gain[i] = self._IG(datas[:,i], labels)
        return np.argmax(gain)

    def fit(self) -> None:
        #find the best node
        #store the node
        #if depth < self.max_depth
            #recurse splitting by the node 
        split = self._splitter(self.trainx, self.trainy)
        depth = 1
        
        root = treeNode(split_name = self.column[split], column_name = np.delete(self.column, split, axis=0), 
                        split = self.trainx[:,split], children = np.delete(self.trainx,split,axis=1), labels = self.trainy,
                        depth = depth, max_depth = self.max_depth)

        def build_tree(node, depth)-> None:
            children_node = []
            depth += 1
            for (temp_x, temp_y) in node.children:
                split = self._splitter(temp_x, temp_y)
                temp = treeNode(split_name = node.column_name[split], column_name = np.delete(node.column_name, split, axis=0), 
                                split = temp_x[:,split], children = np.delete(temp_x,split,axis=1), labels = temp_y, 
                                depth = depth, max_depth = self.max_depth)

                if depth <self.max_depth and not temp.end:
                    temp = build_tree(temp, depth)
                children_node.append(temp)
            node._set_children_node(children_node)
            return node

        self.tree = root if depth==self.max_depth else build_tree(root, depth)
        return
    
    def predict_instance(self, obs, column = None, node = None) -> str:
        #go to node: 
        #   if node.end:
        #       return node.predict_value
        #   else visit child ......
        column = self.column if column is None else column
        temp = self.tree if node is None else node
        index = np.where( column == temp.split_name )
        if temp.end:
            return temp.predict(obs[index])
        visit = temp.visit_children(obs[index])
        if visit == 0: 
            #observation not exist in training data. Happened in car prediction. 
            #One `person' node only have "2" "3" "4" branches, 
            #but test has "5more" 
            return temp.predict(obs[index])
        else:
            return self.predict_instance(np.delete(obs, index, axis=0), column = temp.column_name, node = temp.children_nodes[visit])        

    def predict(self, testx) -> np.ndarray:
        n_sample = testx.shape[0]
        predy = np.empty(n_sample, dtype=object)
        for i in range(n_sample):
            #print(i)
            predy[i] = self.predict_instance(testx[i,:])
        return predy
    