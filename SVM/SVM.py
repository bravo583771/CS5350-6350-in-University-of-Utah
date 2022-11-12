import numpy as np 
from scipy.optimize import minimize

class SVM:
    def __init__(self, trainx: np.ndarray, trainy: np.ndarray, n_epoch: int = 100):
        self.trainx = trainx
        self.trainy = trainy
        self.n_sample, self.n_dim = trainx.shape
        self.n_epoch = n_epoch
    
    def kernel(self, X1, X2, sig = 1):
        #X1 = np.expand_dims(X1, axis = 0).repeat(self.n_sample, axis = 0)
        #X2 = np.expand_dims(X2, axis = 1).repeat(self.n_sample, axis = 1)
        #return np.exp(np.linalg.norm(X1-X2, axis = 2)/sig)
        return np.exp(-np.square(np.linalg.norm(X1[:,None,:]-X2[None,:,:], axis = 2))/sig)

    def optimal_dual(self, C, sig=None, kernel = False):
        A = np.eye(self.n_sample) #idnetical matrix
        C = C*np.ones(self.n_sample)
        y = self.trainy.reshape([-1,1])
        X = self.trainx
        #'jac' is the jacobian matrix of the derivative on alpha
        # alpha y =0
        # A alpha <= C 
        # A alpha >= 0 
        self.constraints = ({'type': 'eq',   'fun': lambda alpha: np.dot(alpha.reshape([-1]), y.reshape([-1])), 'jac': lambda alpha: y.reshape([-1])}, 
                            {'type': 'ineq', 'fun': lambda alpha: C - np.dot(A, alpha.reshape([-1])), 'jac': lambda alpha: -A},
                            {'type': 'ineq', 'fun': lambda alpha: np.dot(A, alpha.reshape([-1])), 'jac': lambda alpha: A}) 

        def dual(alpha):
            alpha = alpha.reshape([-1])            
            #minimize -L = maximize L
            if kernel:
                return 0.5 *  alpha.dot(alpha.dot( self.kernel(X, X, sig) * np.matmul(y, y.transpose()))) - alpha.sum() 
            else:
                return 0.5 * alpha.dot(alpha.dot(np.matmul(X, X.transpose()) * np.matmul(y, y.transpose()))) - alpha.sum() 

        #Partial derivate of Ld on alpha
        def jac_dual(alpha):
            alpha = alpha.reshape([-1])
            if kernel:
                return alpha.dot( self.kernel(X, X, sig) * np.matmul(y, y.transpose())) - np.ones_like(alpha) 
            else:
                return alpha.dot(np.matmul(X, X.transpose()) * np.matmul(y, y.transpose())) - np.ones_like(alpha)
        
        initial = np.zeros([self.n_sample])
        self.optimal = minimize(dual, initial, method = 'SLSQP', jac=jac_dual, constraints=self.constraints)
        opt_alpha = self.optimal.x
        epsilon = 1e-6
        print("the number of support vectors is: ",(opt_alpha > epsilon).sum())
        if kernel:
            #print("the number of support vectors is: ",(opt_alpha > epsilon).sum())
            predict = ((opt_alpha.reshape(-1) * (y.reshape(-1))).reshape([-1])).dot(self.kernel(X, X, sig))
            support_vectors = predict[opt_alpha > epsilon]
            support_labels = y[opt_alpha > epsilon,0]
            b = support_labels[0] - support_vectors[0]
            return opt_alpha, b
        else:
            w = np.matmul(X.transpose(), (opt_alpha.reshape(-1) * (y.reshape(-1))).reshape([-1,1])) 
            support_vectors = X[opt_alpha > epsilon,:]
            support_labels = y[opt_alpha > epsilon,0]
            b = support_labels[0] - np.matmul(support_vectors[0].reshape([1,-1]), w.reshape([-1,1]))
            return np.insert(w,0,b)

    def subGD(self, scheduler, C: float, initial_learning_rate: float = 0.05):
        w = np.zeros([self.n_dim+1])
        for t in range(self.n_epoch):
            learning_rate = scheduler(initial_learning_rate, t)
            shuffle = np.random.choice(np.arange(self.n_sample), self.n_sample, replace=False)
            for i in range(self.n_sample):
                index = shuffle[i] 
                x = np.insert(self.trainx[index,:], 0, 1)
                y = self.trainy[index]
                if np.sum(y*w*x)<=1:
                    w = w-learning_rate*w + learning_rate*C*self.n_sample*y*x
                else:
                    w = (1-learning_rate)*w
        return w
    
    def kernel_perceptron(self, sig = 1):
        c = np.zeros(self.n_sample)
        for t in range(self.n_epoch):
            X = self.trainx
            y = self.trainy
            predict = self.perceptron_predict(X, c, sig = sig)
            c += (predict != y)
        return c

    def perceptron_predict(self, testx, c, sig = 1):
        x1 = np.c_[np.ones(self.n_sample), self.trainx]
        x2 = np.c_[np.ones(testx.shape[0]), testx]
        return 2* (((c.reshape(-1) * (self.trainy.reshape(-1))).reshape([-1])).dot(self.kernel(x1, x2, sig)).reshape(-1)>0)-1

    def predict(self, testx, w = None, alpha = None, b=None, sig = 1, kernal: bool = False):
        if kernal:
            return 2*(((alpha.reshape(-1) * (self.trainy.reshape(-1))).reshape([-1])).dot(self.kernel(self.trainx, testx, sig)).reshape(-1)>0)-1
        else:
            w = w.reshape([-1,1])
            return 2*(np.matmul(np.c_[np.ones(testx.shape[0]), testx], w).reshape(-1)>0)-1
