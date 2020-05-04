import numpy as np
class SVM:
    def __init__(self,alpha,n_epochs,l=1.0,fit_intercept=True):
        self.l=l
        self.alpha=alpha
        self.n_epochs=n_epochs
        self.weights=None
        self.fit_intercept=fit_intercept
        self.intercept=None 
    def fit(self,X,y):  
        self.weights=np.zeros(X.shape[1])
        y=y.reshape(y.shape[0],1)
        X=np.hstack((X,y))
        for e in range(self.n_epochs):
            for ind,_ in enumerate(X):
                if X[ind,-1]*(np.dot(X[ind,:-1],self.weights)+self.intercept)<1:
                    self.weights-=self.alpha*(self.weights*self.l-np.dot(X[ind,-1],X[ind,:-1]))
                    if self.fit_intercept==True:
                        self.intercept-=self.alpha*(-X[ind,-1])
                else:
                    self.weights-=self.alpha*(self.weights*self.l)
            np.random.shuffle(X)
    def partial_fit(self,X,y):
        if type(self.weights)==type(None) and self.fit_intercept==True:
            self.weights=np.zeros(X.shape[1])
            self.intercept=0
        for ind,_ in enumerate(X):
            if y[ind]*(np.dot(X[ind],self.weights)+self.intercept)<1:
                self.weights-=self.alpha*(self.weights*self.l-np.dot(y[ind],X[ind,:]))
                if self.fit_intercept==True:
                    self.intercept-=self.alpha*(-y[ind])
            else:
                self.weights-=self.alpha*(self.weights*self.l)
    def predict(self,X):
        preds=np.dot(X,self.weights)+self.intercept
        return preds
    def hyperplane_coord(self,X):
        y=(-self.intercept-np.dot(X[:,:-1],self.weights[:-1]))/self.weights[-1]
        return y
    def support_vec(self,X,off):
        y=(off-self.intercept-np.dot(X[:,:-1],self.weights[:-1]))/self.weights[-1]
        return y