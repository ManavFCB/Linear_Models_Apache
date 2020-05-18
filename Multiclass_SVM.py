import numpy as np
class MultiSVM:
    def __init__(self,alpha=0.01,tol=1e-3,max_iter=1000,l=1.0,fit_intercept=True):
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
    def fit(self,X,y):
        self.initialize_weights(X,y)
        d=X.shape[1]
        X=np.hstack((X,y.reshape(y.shape[0],1)))
        #curr_error=-np.sum(np.log(probs[np.where(y>=0),y]))
        for e in range(self.max_iter):
            for ind,_ in enumerate(X):
                X_i=X[ind,:-1]
                y_i=X[ind,-1].astype('int')
                score=np.matmul(X_i,self.coefs.T)+self.intercept
                margin=score-score[y_i]+1.0
                margin[y_i]=0
                counts=np.sum(margin>0)
                gradients=np.zeros(margin.shape[0])
                gradients[margin>0]=1
                gradients[y_i]=-counts
                self.coefs-=self.alpha*(np.matmul(gradients.reshape(gradients.shape[0],1),X_i.reshape(1,d))+self.l*self.coefs)
                if self.fit_intercept==True:
                    self.intercept-=self.alpha*(gradients)
            np.random.shuffle(X)
    def partial_fit(self,X,y,classes):
        self.n_classes=classes
        if type(self.coefs)==type(None) and self.fit_intercept==True:
            self.initialize_weights(X,y)
        d=X.shape[1]
        X=np.hstack((X,y.reshape(y.shape[0],1)))
        for ind,_ in enumerate(X):
            X_i=X[ind,:-1]
            y_i=X[ind,-1].astype('int')
            score=np.matmul(X_i,self.coefs.T)+self.intercept
            margin=score-score[y_i]+1.0
            margin[y_i]=0
            counts=np.sum(margin>0)
            gradients=np.zeros(margin.shape[0])
            gradients[margin>0]=1
            gradients[y_i]=-counts
            self.coefs-=self.alpha*(np.matmul(gradients.reshape(gradients.shape[0],1),X_i.reshape(1,d))+self.l*self.coefs)
            if self.fit_intercept==True:
                self.intercept-=self.alpha*(gradients)
    def predict(self,X):
        self.support_values=np.matmul(X,self.coefs.T)+self.intercept
        self.predictions=np.argmax(self.support_values,1)