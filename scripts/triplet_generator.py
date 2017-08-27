import warnings
import numpy as np
import keras.backend as K

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
import multiprocessing
import itertools
from functools import reduce

def make_triplet_loss_func(alpha):
    def triplet_loss(y_true, y_pred):
        # project features on a hyper-sphere.
        anchors = y_pred[0::3]
        positives = y_pred[1::3]
        negatives = y_pred[2::3]
        
        pos_diff = K.sum(K.square(anchors - positives), axis=1)
        neg_diff = K.sum(K.square(anchors - negatives),axis=1)
        loss = K.mean(K.maximum(pos_diff - neg_diff + alpha,0), axis=-1)
        # use y_true to suppress warnings.
        return loss + 0*K.sum(y_true)  
    return triplet_loss

# the following function seems to provide a nice gradient function.
# The code from: https://github.com/maciejkula/triplet_recommendations_keras
def bpr_triplet_loss(y_true, y_pred):
    anchors = y_pred[0::3]
    positives = y_pred[1::3]
    negatives = y_pred[2::3]
    # BPR loss
    loss = K.mean(1.0 - K.sigmoid(
        K.sum(anchors * positives, axis=-1) -
        K.sum(anchors * negatives, axis=-1)), axis=-1)

    return loss - 0 * K.sum(y_true)


def flatten(array):
    return [item for sublist in array for item in sublist]
class TripletGenerator:
    u"""flowを受け取ってtripletにしていくGenerator & loss関数"""
    def __init__(self, base_flow, model, n_jobs=multiprocessing.cpu_count()):
        self.base_flow = base_flow
        self.triplet_buf = None
        self.y_buf = None
        self.model = model
        self.n_jobs = n_jobs
        self.X_stack=[]
        self.y_stack=[]
        
        # initialize model(?)
        #X, _ = next(base_flow)
        #_ = model4triplet.predict(X)

        
    
    def _generator(self,batch_size):
        # get enough number of triplets.
        while self.triplet_buf is None or len(self.triplet_buf) < batch_size:
            X_buf, y_buf = next(self.base_flow)
            X_buf, y_buf = self.get_triplets(X_buf, y_buf)
            if X_buf is None:
                continue
            if self.triplet_buf is None:
                self.triplet_buf = X_buf
                self.y_buf = y_buf
            else:
                self.triplet_buf = np.r_[self.triplet_buf, X_buf]
                self.y_buf = np.r_[self.y_buf, y_buf]
            
        # ensure the batch size to be multiples of 3
        if batch_size%3!=0:
            batch_size = (batch_size//3)*3
            warnings.warn("'batch_size' must be multiples of 3. The batch size is modified to %d"%batch_size)

        # pop batch_size sampels.
        # note that ...
        # X[0::3] : anchors
        # X[1::3] : paired positive sample
        # X[2::3] : paired negative sample
        X = self.triplet_buf[:batch_size]
        y = self.y_buf[:batch_size]
        self.triplet_buf = self.triplet_buf[batch_size:]
        self.y_buf = self.y_buf[batch_size:]
        return X,y
        
    def triplet_flow(self, batch_size=36):
        while True:
            yield self._generator(batch_size)
            

    def get_triplets(self,X,y, embeddings=None):        
        X_trip, y_trip  = self.select_triplets(X, y,embeddings)
        n_triplets = len(X_trip)
        if n_triplets==0:
            warnings.warn("No triplets were obtained!")
            return None,None
        X_trip = np.array(flatten(X_trip))
        y_trip = np.array(flatten(y_trip))
        assert(len(X_trip)/3 == n_triplets)
        assert(len(y_trip)/3 == n_triplets)
        return X_trip, y_trip
    
    def select_triplets(self, X, y, embeddings=None):
        """ Select the triplets for training
        """
        # add stacked samples to the triplet candidates
        if len(self.X_stack)>0:
            X = np.r_[X,self.X_stack]
            y = np.r_[y,self.y_stack]
            self.X_stack=[]
            self.y_stack=[]
        # embeddings can be passed as an argument for debug purpose
        if embeddings is None:
            embeddings = self.model.predict(X)
        return self._select_triplets(embeddings, X, y)

    def _select_triplets(self,embeddings,X,y):        
        num_trips = 0
        triplets = []
        triplets_y = []

        cats = np.unique(y, axis=0)
        dist_mat = pairwise_distances(embeddings, metric='sqeuclidean', n_jobs=self.n_jobs)
        for c in cats: 
            pos_samples = np.where([np.all(a==b) for a,b in zip(y,[c]*len(y))])[0]
            if len(pos_samples)==1:
                # if a sample in category c is isolated, no positive pair is found.
                # to avoid too many missing triplet for category c, stack the sample to the next selection.
                self.X_stack.append(X[pos_samples[0]])
                self.y_stack.append(y[pos_samples[0]])
                continue

            # make triplets for all combination of pos_samples
            for i,j in itertools.combinations(pos_samples,2):
                pos_dist = dist_mat[i,j]
                neg_cands = np.where([np.any(a!=b) for a,b in zip(y,[c]*len(y))] * (dist_mat[i]-pos_dist<self.alpha))[0]
                n_cands = len(neg_cands)
                if n_cands>0:
                    # select negative pair randomly from neg_samples that satisfy neg_dist-pos_dist>=self.alpha.
                    rnd_idx = np.random.choice(n_cands)
                    n_idx = neg_cands[rnd_idx]
                    triplets.append([X[i], X[j], X[n_idx]])
                    triplets_y.append([y[i],y[j],y[n_idx]])
        
        # shuffle
        shuffle_arrays([triplets,triplets_y])
        return triplets, triplets_y
