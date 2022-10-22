import numpy as np 

class linear_regression:
    def __init__(self, trainx: np.ndarray, trainy: np.ndarray, testx: np.ndarray, testy: np.ndarray, initialize = "zero"):
        self.trainx = trainx
        self.trainy = trainy
        self.testx = testx
        self.testy = testy
        self.n_sample, self.n_dim = trainx.shape
        self.optimal = self._optimal()
        #np.random.seed(2022)
        if initialize == "zero":
            self.w = np.zeros([self.n_dim+1, 1])

    def gradiant_w(self, w, x, y):
        n_sample = y.shape[0]
        return -np.matmul(np.c_[np.ones(n_sample), x].transpose(), 
                            (y - np.matmul(np.c_[np.ones(n_sample), x], w).reshape(-1)).reshape([-1,1]))
    
    def loss(self, x, y):
        n_sample = y.shape[0]
        return np.sum(0.5*np.square(y - np.matmul(np.c_[np.ones(n_sample), x], self.w).reshape(-1)))

    def SGD(self, max_iter = 100000, learning_rate = 0.001, converge_criteria = 10**-6):
        loss = []
        test_loss = []
        n_sample = 1
        if n_sample >=self.n_sample:
            raise ValueError("Sample size is only {}.".format(self.n_sample))
        for _ in range(max_iter):
            index = np.random.choice(np.arange(self.n_sample), n_sample, replace=False)
            x = self.trainx[index,:].reshape( [n_sample, -1] )
            y = self.trainy[index]
            #np.insert(x, 0, 1)
            gradiant_w = -np.matmul(np.c_[np.ones(n_sample), x].transpose(), 
                                    (y - np.matmul(np.c_[np.ones(n_sample), x], self.w).reshape(-1)).reshape([-1,1]))
            #gradiant_w = self.gradiant_w(self.w, x, y)
            new_w = self.w - learning_rate * gradiant_w
            #learning_rate = 0.999 * learning_rate
            if np.linalg.norm(new_w - self.w) < converge_criteria:
                self.w = new_w
                loss.append(self.loss(self.trainx, self.trainy))
                test_loss.append(self.loss(self.testx, self.testy))
                break
            else:
                self.w = new_w
                loss.append(self.loss(self.trainx, self.trainy))
                test_loss.append(self.loss(self.testx, self.testy))
        return loss, test_loss

    def bGD(self, max_iter = 100000, learning_rate = 0.001, converge_criteria = 10**-6):
        loss = []
        test_loss = []
        for _ in range(max_iter):
            gradiant_w = -np.matmul(np.c_[np.ones(self.n_sample), self.trainx].transpose(), 
                                    (self.trainy - np.matmul(np.c_[np.ones(self.n_sample), self.trainx], self.w).reshape(-1)).reshape([-1,1]))/self.n_sample
            #gradiant_w = self.gradiant_w(self.w, self.trainx, self.trainy)
            new_w = self.w - learning_rate * gradiant_w
            if np.linalg.norm(new_w - self.w) < converge_criteria:
                self.w = new_w
                loss.append(self.loss(self.trainx, self.trainy))
                test_loss.append(self.loss(self.testx, self.testy))
                break
            else:
                self.w = new_w
                loss.append(self.loss(self.trainx, self.trainy))
                test_loss.append(self.loss(self.testx, self.testy))
        return loss, test_loss
    
    def _optimal(self,):
        x = np.c_[np.ones(self.n_sample), self.trainx].transpose()
        y = self.trainy
        return np.matmul(np.matmul(np.linalg.inv(np.matmul(x, x.transpose())), x), y)
    
    def predict(self, testx):
        return np.matmul(np.c_[np.ones(self.n_sample), testx], self.w)