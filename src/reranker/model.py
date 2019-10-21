from abc import ABC, abstractmethod
import pandas as pd
import random
 
class RankerInterface(ABC):
    """
    This abstract class must be implemented by all training models
    """
 
    def __init__(self, featureengineer):
        self.fe = featureengineer 
        self.predictions = pd.DataFrame(columns = ['q_num', 'qid', 'doc_id', 'rank'])  
        super().__init__()


  
    @abstractmethod
    def train(self, inputhandler):
        """
        uses the labelled queries from the inputhandler to estimate a ranking model
        """
        pass


    def predict(self, inputhandler):
        """
        uses the query sequences from the inputhandler to rerank the documents according to the trained model. 
        must return a dataframe with columns [sid, q_num, qid, doc_id, rank]
        """
        self.predictions = self._predict(inputhandler)[['sid', 'q_num', 'qid', 'doc_id', 'rank']]
        return self.predictions

   
    @abstractmethod
    def _predict(self, inputhandler):        
        pass


class RandomRanker(RankerInterface):

    def __shuffle_group(self, group):
        group.loc[:, 'doc_id'] = random.sample(group['doc_id'].to_list(), len(group['doc_id']))
        group.loc[:, 'rank'] = range(1, len(group['doc_id'])+1)
        return group

    def _predict(self, inputhandler):
        pred = inputhandler.get_query_seq().groupby(['sid', 'q_num', 'qid']).apply(self.__shuffle_group)
        return pred

    def train(self, inputhandler):
        pass
