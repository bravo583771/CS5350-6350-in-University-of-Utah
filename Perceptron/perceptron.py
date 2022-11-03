import numpy as np 

class perceptron:
    def __init__(self, trainx: np.ndarray, trainy: np.ndarray, n_epoch: int = 10):
        self.trainx = trainx
        self.trainy = trainy
        self.n_sample, self.n_dim = trainx.shape
        self.n_epoch = n_epoch
    
    def standard(self, learning_rate: float = 0.05):
        w = np.zeros([self.n_dim+1, 1])
        for _ in range(self.n_epoch):
            index = np.random.choice(np.arange(self.n_sample), self.n_sample, replace=False)
            x = self.trainx[index,:].reshape( [self.n_sample, -1] )
            y = self.trainy[index]
            predy = np.matmul(np.c_[np.ones(self.n_sample), x], w).reshape(-1)
            for i in range(self.n_sample):
                if predy[i]*y[i]<=0:
                    w += learning_rate*y[i]*(np.c_[np.ones(self.n_sample), x][i,:]).reshape([-1,1])
        return w
    
    def voted(self, learning_rate: float = 0.05):
        w = np.zeros([self.n_dim+1, 1])
        ws = []
        m = 0
        c = 0
        cs = []
        for _ in range(self.n_epoch):
            index = np.random.choice(np.arange(self.n_sample), self.n_sample, replace=False)
            x = self.trainx[index,:].reshape( [self.n_sample, -1] )
            y = self.trainy[index]
            predy = np.matmul(np.c_[np.ones(self.n_sample), x], w).reshape(-1)
            for i in range(self.n_sample):
                if m==0:
                    w += learning_rate*y[i]*(np.c_[np.ones(self.n_sample), x][i,:]).reshape([-1,1])
                    m += 1
                    c = 1
                    continue
                if predy[i]*y[i]<=0:
                    ws.append(np.copy(w))
                    w += learning_rate*y[i]*(np.c_[np.ones(self.n_sample), x][i,:]).reshape([-1,1])
                    m += 1
                    cs.append(c)
                    c = 1
                else:
                    c += 1
        return ws, cs
    
    def average(self, learning_rate: float = 0.05):
        w = np.zeros([self.n_dim+1, 1])
        a = np.zeros([self.n_dim+1, 1])
        for _ in range(self.n_epoch):
            index = np.random.choice(np.arange(self.n_sample), self.n_sample, replace=False)
            x = self.trainx[index,:].reshape( [self.n_sample, -1] )
            y = self.trainy[index]
            predy = np.matmul(np.c_[np.ones(self.n_sample), x], w).reshape(-1)
            for i in range(self.n_sample):
                if predy[i]*y[i]<=0:
                    w += learning_rate*y[i]*(np.c_[np.ones(self.n_sample), x][i,:]).reshape([-1,1])
                a += np.copy(w)
        return a

    def predict(self, testx, w, ws = None, cs = None):
        n_sample = testx.shape[0]
        if cs is None:
            final_predict = np.ones(n_sample)
            predy = np.matmul(np.c_[np.ones(n_sample), testx], w).reshape(-1)
            final_predict[predy<0] = -1
            return final_predict
        else:
            final_predict = np.zeros(n_sample)
            for wi, ci in zip(ws, cs):
                #print(ci)
                temp_predict = np.ones(n_sample)
                predy = ci * np.matmul(np.c_[np.ones(n_sample), testx], wi).reshape(-1)
                temp_predict[predy<0] = -1
                final_predict += temp_predict
            final_predict[final_predict>=0]=1
            final_predict[final_predict<0]=-1
            return final_predict
