import numpy as np
import pandas as pd



class Evaluator:

    def __init__(self, inputhandler):
        self.inputhandler = inputhandler

    
    def __DCG(self, rel, p):
        dcg = rel[:p]/np.log2(range(2, p+2))
        return np.sum(dcg) 


    def __nDCG(self, rel, p):
        dcg = self.__DCG(rel, p)
        idcg = self.__DCG(np.flip(np.sort(rel)), p)
        return dcg/idcg

    def __err(self, rel, gamma):
        n = len(rel)
        p = 1
        ERR = 0
        rel = rel.tolist()
        for r in range(1, n, 1):
            R = (2**rel[r-1])/2
            ERR += p*R/r
            p *= (1-R)
        return ERR


    def mean_ndcg(self, pred):
        pred = pd.merge(pred, self.inputhandler.get_queries()[['qid', 'doc_id','relevance']].drop_duplicates(), 
                        how = 'left', on = ['qid', 'doc_id'])
        pred.sort_values(['sid', 'q_num', 'rank'], inplace = True)
        ndcg = pred.groupby(['sid', 'q_num']).relevance.apply(lambda x: self.__nDCG(x, len(x)))
        return ndcg.groupby('sid').mean()


    def ERR(self, pred):
        pred = pd.merge(pred, self.inputhandler.get_queries()[['qid', 'doc_id','relevance']].drop_duplicates(), 
                        how = 'left', on = ['qid', 'doc_id'])
        pred.sort_values(['sid', 'q_num', 'rank'], inplace = True)
        err = pred.groupby(['sid', 'q_num']).relevance.apply(lambda x: self.__err(x, 0.8))
        return err.groupby('sid').mean()




    






    

    





            





