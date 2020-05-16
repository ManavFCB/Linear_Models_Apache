import numpy as np
class MLR:
    def __init__(self,alpha=0.05,tol=1e-3,max_iter=500,l=0.0,fit_intercept=True):
        self.alpha=alpha
        self.tol=tol
        self.max_iter=max_iter
        self.l=l
        self.fit_intercept=fit_intercept
        self.coefs=None
        self.intercept=None
        self.n_classes=None
    def initialize_weights(self,X,y):
        if type(self.n_classes)==type(None):
            self.n_classes=np.unique(y).shape[0]
        self.coefs=np.zeros((self.n_classes,X.shape[1]))
        if self.fit_intercept==True:
            self.intercept=np.zeros(self.n_classes)
    def softmax(self,terms):
        probs=terms/np.sum(terms)
        return probs
    def identifier(self,p,n):   #this returns 1 if y==k (i.e if the value of y is the class of interest) and 0 otherwise
        y_ind=np.zeros(n)
        y_ind[p]=1
        return y_ind
    def fit(self,X,y):
        self.initialize_weights(X,y)
        self.idents=np.array([self.identifier(p,self.n_classes) for p in y])
        d=X.shape[1]
        X=np.hstack((self.idents,X))
        #curr_error=-np.sum(np.log(probs[np.where(y>=0),y]))
        for e in range(self.max_iter):
            for ind,_ in enumerate(X):
                idents=X[ind,:self.n_classes]
                X_i=X[ind,self.n_classes:]
                inp_soft=np.exp(np.matmul(X_i,self.coefs.T)+self.intercept)
                probs=self.softmax(inp_soft)
                self.coefs+=self.alpha*(np.matmul(idents.reshape(self.idents.shape[1],1)-probs.reshape(probs.shape[0],1),X_i.reshape(1,d))+self.l*self.coefs)
                if self.fit_intercept==True:
                    self.intercept+=self.alpha*(idents-probs)
            np.random.shuffle(X)
        X=X[:,self.n_classes:]
    def partial_fit(self,X,y,classes):
        self.n_classes=classes
        if type(self.coefs)==type(None) and self.fit_intercept==True:
            self.initialize_weights(X,y)
        d=X.shape[1]
        self.idents=np.array([self.identifier(p,self.n_classes) for p in y])
        #idx=np.random.choice(np.arange(0,X.shape[0]),X.shape[0],replace=False)
        for ind,_ in enumerate(X):
            idents=self.idents[ind]
            X_i=X[ind]
            inp_soft=np.exp(np.matmul(X_i,self.coefs.T)+self.intercept)
            probs=self.softmax(inp_soft)
            self.coefs+=self.alpha*(np.matmul(idents.reshape(self.idents.shape[1],1)-probs.reshape(probs.shape[0],1),X_i.reshape(1,d))+self.l*self.coefs)
            if self.fit_intercept==True:
                self.intercept+=self.alpha*(idents-probs)
    def predict(self,X):
        inp_soft=np.exp(np.matmul(X,self.coefs.T)+self.intercept)
        probs=self.softmax(inp_soft)
        self.predictions=np.argmax(probs,1)

