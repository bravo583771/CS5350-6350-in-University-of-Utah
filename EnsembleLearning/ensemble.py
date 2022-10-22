import numpy as np 
import copy
#import pandas as pd

class Node(object):
    def __init__(self, split_name, column_name, split: np.ndarray, children: np.ndarray, labels: np.ndarray, weight) -> None:
        self.split_name = split_name #column name of this node, like 'safe'
        self.column_name = column_name #column names of children
        self.predict_value = []
        #self.children = [] #store (trainx, trainy) subsets
        self.children_value = np.unique(split) #the class values of split_name, like 'safe' = ['low','med','high']
        for v in self.children_value:
            #self.children.append((children[split == v,:], labels[split == v]))
            y = labels[split == v]
            w = weight[split == v]
            yvalue = np.unique(y)
            probs = np.array([w[y == vy].sum() for vy in yvalue])
            probs = probs/probs.sum()

            self.predict_value.append(yvalue[np.argmax( probs)])
            #print("for {} = {}, majority label is {}({})".format(self.split_name, v, value[np.argmax( counts)], counts[np.argmax( counts)]/counts.sum()))
            #when the tree is end due to any reasons, the predict value will be the majority class y at this branch.
        yvalue = np.unique(labels)
        probs = np.array([weight[labels == v].sum() for v in yvalue])
        probs = probs/probs.sum()
        self.predict_value.append(yvalue[np.argmax( probs)]) #for x value not in self.children_value
            #observation not exist in training data. Happened in car prediction. 
            #A `person' node only has "2" "3" "4" branches, 
            #but test observation is "5more" 

    def predict(self, x):
        if x not in self.children_value:
            return self.predict_value[-1]
        return self.predict_value[np.argwhere(self.children_value == x).item()]

class DecisionStump(object):
    def __init__(self, trainx: np.ndarray, trainy: np.ndarray, column: np.ndarray, criterion: str = "entropy", entropy_base = None) -> None:
        if criterion not in ["entropy"]:
            raise ValueError("criterion should be \"entropy\"")
        self.criterion = criterion #"entropy", "gini", "me"
        self.tree = None
        self.trainx = trainx
        self.trainy = trainy
        self.column = column #columns name
        self.base = entropy_base #log base
        if self.criterion == "entropy":
            self.H = self._entropy

    def _entropy(self, labels: np.ndarray, weight: np.ndarray) -> np.float32:
        # compute entropy
        yvalue = np.unique(labels)
        probs = np.array([weight[labels == v].sum() for v in yvalue])
        probs = probs/probs.sum()
        return -(probs * np.log(probs)).sum() if self.base is None else -(probs * np.log(probs)/np.log(self.base)).sum()

    def _IG(self, x: np.ndarray, labels: np.ndarray, weight: np.ndarray) -> np.float32:
        #compute information gain
        IG = self.H(labels, weight)
        values = np.unique(x)
        pxs = np.array([weight[x == v].sum() for v in values])
        pxs = pxs/pxs.sum()
        #print(pxs)
        #values, counts = np.unique(x, return_counts=True)
        #pxs = counts / np.size(x)
        for (value, px) in zip(values, pxs):
            IG -= px * self.H(labels[x == value], weight[x == value])
        return IG
    
    def _splitter(self, datas: np.ndarray, labels: np.ndarray, weight: np.ndarray) -> np.int :
        n_feature = datas.shape[1]
        gain = np.zeros(n_feature)
        for i in range(n_feature):
            gain[i] = self._IG(datas[:,i], labels, weight)
        #print(gain)
        #print(gain[np.argmax(gain)])
        return np.argmax(gain)

    def fit(self, weight) -> None:
        #find the best node
        #store the node
        #if depth < self.max_depth
            #recurse splitting by the node 
        #print("\n")
        #print(self.column)
        split = self._splitter(self.trainx, self.trainy, weight)
        #print(split)
        self.tree = Node(split_name = self.column[split], column_name = np.delete(self.column, split, axis=0), 
                        split = np.squeeze(self.trainx[:,split]), children = np.squeeze(np.delete(self.trainx,split,axis=1)), 
                        labels = np.squeeze(self.trainy), weight = weight
                        )
        self.predy = self.predict(self.trainx)
        return
    
    def predict_instance(self, obs: np.ndarray, column = None, node = None) -> str:
        #go to node: 
        #   if node.end:
        #       return node.predict_value
        #   else visit child ......
        column = self.column if column is None else column
        temp = self.tree if node is None else node
        index = np.where( column == temp.split_name )
        return temp.predict(obs[index])       

    def predict(self, testx: np.ndarray) -> np.ndarray:
        if self.tree is None:
            raise RuntimeError("Please fit the model before predicting.")
        n_sample = testx.shape[0]
        predy = np.empty(n_sample, dtype=object)
        for i in range(n_sample):
            #print(i)
            predy[i] = self.predict_instance(testx[i,:])
        return predy

    def remove_train_data(self):
        self.trainx, self.trainy = None, None


class AdaBoost(object):
    def __init__(self, trainx: np.ndarray, trainy: np.ndarray, column: np.ndarray, entropy_base = None) -> None:
        self.trainx = trainx
        self.trainy = trainy
        self.column = column #columns name
        self.base = entropy_base #log base

        self.n_sample = trainy.shape[0]
        self.D = np.ones(self.n_sample, dtype = np.float64)/self.n_sample
        self.alpha = []
        self.h = []
        #np.random.seed(2022)
        self.yvalue = np.unique(trainy)
        self.ymap = dict()
        self.ymap[self.yvalue[0]]=-1
        self.ymap[self.yvalue[1]]=1

    
    def fit(self, max_iter: int=500):
        for iter in range(1,max_iter+1):
            print("Adaboost fitting, iteration: {}".format(iter))
            #print(self.D)
            mytree = DecisionStump(trainx = self.trainx, trainy = self.trainy, column = self.column, criterion = "entropy", entropy_base = self.base)
            mytree.fit(self.D)
            #print(self.mytree.predy, self.mytree.trainy)
            incorrect = mytree.predy != mytree.trainy 
            h_times_y = np.ones(self.n_sample)
            h_times_y[incorrect] = -1
            #print(h_times_y)
            #epsilon = 0.5-0.5*np.sum(self.D*h_times_y)   
            epsilon = np.average(incorrect, weights = self.D)       
            
            #if epsilon>0.5:
            #    print("error > 0.5 at iter={}".format(iter))
            #    break
            
            if epsilon== 0:
                print("error = 0 at iter={}".format(iter))
                break
            
            store_tree = copy.deepcopy(mytree)
            store_tree.remove_train_data()
            self.h.append(store_tree)
            del store_tree
            
            #print(epsilon)
            alpha_t = 0.5*np.log((1-epsilon)/epsilon)
            self.alpha.append(alpha_t)
            #print(alpha_t)
            self.D = self.D*np.exp(-alpha_t*h_times_y)
            #self.D = np.exp(np.log(self.D) + alpha_t * incorrect * (self.D > 0))
            #print(self.D)
            
            self.D = self.D/(self.D.sum())
            #print(self.D) 
        #self.predy = self.predict(self.mytree.trainx)

    #def predict(self, testx: np.ndarray, iter: int = 500) -> np.ndarray:
    def predict(self, testx: np.ndarray) -> np.ndarray:
        n_sample = testx.shape[0]
        predy = np.zeros(n_sample)
        n_models = len(self.h)
        if n_models<500:
            print("The training procedure stop at iter{}.".format(n_models))
        #print(n_models)
        stump = []
        predict = []
        for i in range(n_models):
            print("predict y using {} of weak learners".format(i+1))
            h, alpha = self.h[i], self.alpha[i]
            tempy = h.predict(testx)
            stump.append(tempy)
            tempy = np.vectorize(self.ymap.get)(tempy)
            #print(tempy)
            predy += alpha*tempy

            final_y = np.empty(n_sample, dtype=object)
            final_y[predy<0] = self.yvalue[0]
            final_y[predy>=0] = self.yvalue[1]
            predict.append(final_y.copy())
        return predict, stump
        


class treeNode(object):
    def __init__(self, split_name, column_name, split: np.ndarray, children: np.ndarray, labels: np.ndarray, depth: int, max_depth: int=6) -> None:
        self.split_name = split_name #column name of this node, like 'safe'
        self.column_name = column_name #column names of children
        self.end = True if np.unique(labels).shape[0] == 1 or depth==max_depth else False
        self.depth = depth

        self.predict_value = []
        self.children = [] #store (trainx, trainy) subsets
        self.children_value = np.unique(split) #the class values of split_name, like 'safe' = ['low','med','high']
        for v in self.children_value:
            self.children.append((children[split == v,:], labels[split == v]))
            value, counts = np.unique(labels[split == v], return_counts=True)
            self.predict_value.append(value[np.argmax( counts)])
            #print("for {} = {}, majority label is {}({})".format(self.split_name, v, value[np.argmax( counts)], counts[np.argmax( counts)]/counts.sum()))
            #when the tree is end due to any reasons, the predict value will be the majority class y at this branch.
        value, counts = np.unique(labels, return_counts=True)
        self.predict_value.append(value[np.argmax( counts)]) #for x value not in self.children_value
            #observation not exist in training data. Happened in car prediction. 
            #A `person' node only has "2" "3" "4" branches, 
            #but test observation is "5more" 
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
    def __init__(self, trainx: np.ndarray, trainy: np.ndarray, column: np.ndarray, criterion: str = "entropy", max_depth: int=16, entropy_base = None) -> None:
        if criterion not in ["entropy", "gini", "me"]:
            raise ValueError("criterion should be \"entropy\", \"gini\", or \"me\"")
        if max_depth<1 or not isinstance(max_depth, int):
            raise ValueError("max_depth should be a positive integer.")
        self.criterion = criterion #"entropy", "gini", "me"
        self.max_depth = min(column.shape[0], max_depth) #should be not larger than column.shape[0]
        self.tree = None
        self.trainx = trainx
        self.trainy = trainy
        self.column = column #columns name
        self.base = entropy_base #log base
        if self.criterion == "entropy":
            self.H = self._entropy
        elif self.criterion == "gini":
            self.H = self._gini
        elif self.criterion == "me":
            self.H = self._ME

    def _entropy(self, labels: np.ndarray) -> np.float32:
        # compute entropy
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / np.size(labels)
        return -(probs * np.log(probs)).sum() if self.base is None else -(probs * np.log(probs)/np.log(self.base)).sum()

    def _ME(self, labels: np.ndarray) -> np.float32:
        #compute majority error
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / np.size(labels)
        return 1-probs.max() 

    def _gini(self, labels: np.ndarray) -> np.float32:
        #compute gini index
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / np.size(labels)
        return 1-np.square(probs).sum() 

    def _IG(self, x: np.ndarray, labels: np.ndarray) -> np.float32:
        #compute information gain
        IG = self.H(labels)
        values, counts = np.unique(x, return_counts=True)
        pxs = counts / np.size(x)
        for (value, px) in zip(values, pxs):
            IG -= px * self.H(labels[x == value])
        return IG
    
    def _splitter(self, datas: np.ndarray, labels: np.ndarray) -> None :
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
        def build_tree(node: treeNode, depth: int)-> treeNode:
            
            children_node = []
            #depth += 1
            for (temp_x, temp_y) in node.children:
                split = self._splitter(temp_x, temp_y)
                temp = treeNode(split_name = node.column_name[split], column_name = np.delete(node.column_name, split, axis=0), 
                                split = temp_x[:,split], children = np.delete(temp_x, split, axis=1), labels = temp_y, 
                                depth = depth+1, max_depth = self.max_depth)

                if depth <self.max_depth-1 and not temp.end:
                    temp = build_tree(temp, depth+1)
                children_node.append(temp)
            node._set_children_node(children_node)
            return node

        self.tree = root if depth==self.max_depth else build_tree(root, depth)
        return
    
    def predict_instance(self, obs: np.ndarray, column = None, node = None) -> str:
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

    def predict(self, testx: np.ndarray) -> np.ndarray:
        n_sample = testx.shape[0]
        predy = np.empty(n_sample, dtype=object)
        for i in range(n_sample):
            #print(i)
            predy[i] = self.predict_instance(testx[i,:])
        return predy


class Bagging(object):
    def __init__(self, trainx: np.ndarray, trainy: np.ndarray, column: np.ndarray, max_depth: int=16) -> None:
        self.n_sample = trainy.shape[0]
        self.tree = []
        #np.random.seed(2022)
        self.trainx = trainx
        self.trainy = trainy
        self.column = column
        self.max_depth = min(column.shape[0], max_depth)
        self.entropy_base = self.max_depth 

        self.yvalue = np.unique(trainy)
        self.ymap = dict()
        self.ymap[self.yvalue[0]]=-1
        self.ymap[self.yvalue[1]]=1

    
    def fit(self, max_trees: int=500):
        for iter in range(1, max_trees+1):
            print("Bagging fitting, subtree: {}".format(iter))
            #print(self.D)
            index = np.random.choice(np.arange(self.n_sample), size=self.n_sample, replace=True)
            trainx = self.trainx[index,:]
            trainy = self.trainy[index]
            mytree = DecisionTree(trainx = trainx, trainy = trainy, column = self.column, criterion = "entropy", 
                                  max_depth = self.max_depth, entropy_base = self.entropy_base)
            mytree.fit()
            self.tree.append(mytree)

    #def predict(self, testx: np.ndarray, iter: int = 500) -> np.ndarray:
    def predict(self, testx: np.ndarray) -> np.ndarray:
        n_sample = testx.shape[0]
        predy = np.zeros(n_sample)
        n_models = len(self.tree)
        if n_models<500:
            print("The training procedure stop at iter{}.".format(n_models))
        #print(n_models)
        predict = []
        for i in range(n_models):
            print("predict y using {} of bagging trees".format(i+1))
            tree = self.tree[i]
            tempy = tree.predict(testx)
            tempy = np.vectorize(self.ymap.get)(tempy)
            predy += tempy

            final_y = np.empty(n_sample, dtype=object)
            final_y[predy<0] = self.yvalue[0]
            final_y[predy>=0] = self.yvalue[1]
            predict.append(final_y.copy())
        return predict

class RandomForest(object):
    def __init__(self, trainx: np.ndarray, trainy: np.ndarray, column: np.ndarray, max_depth: int=16, select_features: int=6) -> None:
        self.n_sample = trainy.shape[0]
        self.n_feature = column.shape[0]
        self.tree = []
        #np.random.seed(2022)
        self.trainx = trainx
        self.trainy = trainy
        self.column = column
        self.select_features = select_features
        self.max_depth = min(column.shape[0], max_depth)
        self.entropy_base = self.max_depth 

        self.yvalue = np.unique(trainy)
        self.ymap = dict()
        self.ymap[self.yvalue[0]]=-1
        self.ymap[self.yvalue[1]]=1

    
    def fit(self, max_trees: int=500):
        self.selects = []
        for iter in range(1, max_trees+1):
            print("RandomForest fitting, subtree: {}".format(iter))
            #print(self.D)
            index = np.random.choice(np.arange(self.n_sample), size=self.n_sample, replace=True)
            select_feature = np.random.choice(np.arange(self.n_feature), size=self.select_features, replace=False)
            self.selects.append(select_feature)
            trainx = self.trainx[:, select_feature]
            trainx = trainx[index,:]
            trainy = self.trainy[index]
            mytree = DecisionTree(trainx = trainx, trainy = trainy, column = self.column[select_feature], criterion = "entropy", 
                                  max_depth = self.max_depth, entropy_base = self.entropy_base)
            mytree.fit()
            self.tree.append(mytree)

    #def predict(self, testx: np.ndarray, iter: int = 500) -> np.ndarray:
    def predict(self, testx: np.ndarray) -> np.ndarray:
        n_sample = testx.shape[0]
        predy = np.zeros(n_sample)
        n_models = len(self.tree)
        if n_models<500:
            print("The training procedure stop at iter{}.".format(n_models))
        #print(n_models)
        predict = []
        for i in range(n_models):
            select_feature = self.selects[i]
            print("predict y using {} of bagging trees".format(i+1))
            tree = self.tree[i]
            tempy = tree.predict(testx[:,select_feature])
            tempy = np.vectorize(self.ymap.get)(tempy)
            predy += tempy

            final_y = np.empty(n_sample, dtype=object)
            final_y[predy<0] = self.yvalue[0]
            final_y[predy>=0] = self.yvalue[1]
            predict.append(final_y.copy())
        return predict