import numpy as np 
import pickle
class linUCB:
    def __init__(self, n_arms, n_feature=5,alpha=1.0 ):
        self.n_arms=n_arms
        self.n_feature=n_feature
        self.alpha=alpha
        self.A= [np.identity(n_feature) for _ in range(n_arms)] #ma tran thong tin
        self.b= [np.zeros(n_feature) for _ in range(n_arms)] #vector reward
    def warm_start(self, history, seed=0, n_sampling=3):
        rng = np.random.default_rng(seed)
        for x, bought in history:
            bought = set(bought)
            for a in bought:
                self.update(a, x, 1.0)
            others = [a for a in range(self.n_arms) if a not in bought]
            for a in rng.choice(others, size=min(n_sampling, len(others)), replace=False):
                self.update(a, x, 0.0)
    def get_score(self, feature_vector):
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
        return np.array(scores)
    def select_arm(self,k, feature_vector):
        scores=self.get_score(feature_vector)
        noise=np.random.uniform(0, 1e-9,self.n_arms)
        order=np.argsort(-(noise+scores))[:k]
        return order,scores[order]

    def update(self, arm, context, reward):
        x=np.asarray(context, dtype=float)
        self.A[arm] += np.outer(x,x)
        self.b[arm] += reward *x
    def save(self, path):
        with open(path,"wb") as f:
            pickle.dump(self,f)
    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
