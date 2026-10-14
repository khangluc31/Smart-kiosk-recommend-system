import numpy as np 
class linUCB:
    def __init__(self, n_arms, n_feature=5,alpha=1.0 ):
        self.n_arms=n_arms
        self.n_feature=n_feature
        self.alpha=alpha
        self.A= [np.identity(n_feature) for _ in range(n_arms)] #ma tran thong tin
        self.b= [np.zeros(n_feature) for _ in range(n_arms)] #vector reward
    def select_arm(self, feature_vector):
        features=np.asarray(feature_vector)
        scores=[]
        for a in range(self.n_arms):
            x=features[a]
            A_inv=np.linalg.inv(self.A[a])
            theta=A_inv@self.b[a]
            exploitation=theta.T@x
            exploration=self.alpha*np.sqrt(x.T@A_inv@x)
            point=exploitation+exploration
            scores.append(point)
        return np.argsort(scores)[-5:][::-1]
    def update(self, arm, context, reward):
        x=np.asarray(context, dtype=float)
        self.A[arm] += np.outer(x,x)
        self.b[arm] += reward *x
