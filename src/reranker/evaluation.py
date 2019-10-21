from multiprocessing import Process, Manager
import pandas as pd
import numpy as np



class Evaluator:
  
    gamma = 0.8  

    def __init__(self, inputhandler):
        self.inputhandler = inputhandler

        
    def f(self, relevance):
        # monotonic transformation of relevance into probability
        return 1 / (1 + np.exp(-relevance))


    def add_prob(self, x: pd.DataFrame):
        cum_rel = np.cumprod(1 -f(x.groupby('rank').max().relevance))
        return [gamma ** (row['rank'] - 1) * cum_rel.get(row['rank']) for (i,row) in x.iterrows()]

    
    def __DCG(self, rel, p):
        dcg = rel[:p]/np.log2(range(2, p+2))
        return np.sum(dcg) 


    def __nDCG(self, rel, p):
        dcg = self.__DCG(rel, p)
        idcg = self.__DCG(np.flip(np.sort(rel)), p)
        return dcg/idcg


    def mean_ndcg(self, pred):
        pred = pd.merge(pred, self.inputhandler.get_queries()[['qid', 'doc_id','relevance']].drop_duplicates(), 
                        how = 'left', on = ['qid', 'doc_id'])
        pred.sort_values(['q_num', 'rank'], inplace = True)
        ndcg = pred.groupby('q_num').relevance.apply(lambda x: self.__nDCG(x, len(x)))
        return(ndcg.mean())
      
    def exposures_and_relevances(self, pi):
        prob_rel = pi['relevance'].apply(lambda x: f(x))
        return pd.Series({'exp': sum(pi['prob']), 'rel': sum(prob_rel)})


    def delta(self, PI):
        series = PI.groupby('qid').apply(self.add_prob)
        prob = []
        for qid in pd.unique(PI['qid']):
            prob.append(series[qid])

        PI['prob'] = [item for sublist in prob for item in sublist]

        exp_and_rel = PI.groupby(['gid', 'qid', 'author_id'])\
            .apply(self.exposures_and_relevances)\
            .groupby(['gid', 'author_id']).sum()\
            .groupby('gid').sum()

        group_exposures = exp_and_rel['exp'] / exp_and_rel['exp'].values.sum()
        group_relevances = exp_and_rel['rel'] / exp_and_rel['rel'].values.sum()

        return np.square(sum(np.square(group_exposures - group_relevances)))
