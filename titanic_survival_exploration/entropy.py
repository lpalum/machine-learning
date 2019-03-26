
import numpy as np
import math

def entropy(seq, base=None):
  """ Computes entropy of sequence distribution. - 
      information entropy is the average amount of information conveyed by any 
      event (random draw from set), when considering all possible outcomes.
  
      args
      ----
      seq : list
      base : optional base 2 if not specified
      
      returns
      -------
      ent : floating point value between 0 and 1
  """
  
  # length of the information sequence
  seq_len = len(seq)

  # Empty or one value has an entropy of 1 by default (no impurity)
  if seq_len <= 1:
    return 0

  # Calcuate the number of unique values in the sequence and what their respective probability is freq/total length
  value,counts = np.unique(seq, return_counts=True)
  probs = counts / seq_len
  n_classes = np.count_nonzero(probs)

  # if there is only one class of data in the sequence there is no entropy (no impurity)
  if n_classes <= 1:
    return 0

  ent = 0.

  # Compute entropy
  base = 2 if base is None else base
  
  return np.sum([(i * math.log(i, base))*-1 for i in probs])

def information_gain(pent, splits, seq):
    """
    Information Gain - is the change in entropy associated with any new 
    information (restricting the distribution)
    
    args
    ----
    pent : float parent seqence entropy.
    splits : list of boolean filters for the squence 
    seq : Pandas dataframe with the target value and a filtering feature.
    
    returns
    -------
    gain : float the reduction in entropy of the target variable based on the filtering feature.
    
    """
    # length of the information sequence
    seq_len = len(seq)

    # Empty or one value has an entropy of 1 by default (no impurity)
    if seq_len <= 1:
        return pent

    # Calcuate the number of unique values in the sequence and what their respective probability is freq/total length
    counts = [len(seq[x]['Survived']) for x in splits]
    probs = [x/seq_len for x in counts] 
    n_classes = np.count_nonzero(probs)
    
    # if there is only one class of data in the sequence there is no entropy (no impurity)
    if n_classes <= 1:
        return pent
    
    ents = [entropy(seq[x]['Survived']) for x in splits]
   
    return pent - np.dot(probs,ents)


