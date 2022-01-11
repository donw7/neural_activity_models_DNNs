import numpy as np
from sklearn import linear_model

# get k-fold splits
def get_test_train_splits(data, decisions, n_folds=5):
  """
  Returns a tuple of matched train sets and validation sets, rotating through the data.
  
  """
  
  fold_size = len(data)//n_folds
  
  training_X = [np.roll(data,fold_size*i, axis=0)[fold_size:] for i in range(n_folds)]
  val_X = [np.roll(data,fold_size*i, axis=0)[:fold_size] for i in range(n_folds)]
  
  training_Y = [np.roll(decisions,fold_size*i, axis=0)[fold_size:] for i in range(n_folds)]
  val_Y = [np.roll(decisions,fold_size*i, axis=0)[:fold_size] for i in range(n_folds)]
  

  return (training_X, training_Y), (val_X, val_Y)
  
def get_kfold_validation_score(data, decisions, C):
  
  (training_X, training_Y), (val_X, val_Y) = get_test_train_splits(data, decisions)

  scores = []

  for fold in range(5):
    training_Xi = training_X[fold]
    training_Yi = training_Y[fold]

    val_Xi = val_X[fold]
    val_Yi = val_Y[fold]

    # re-initialize each iteration
    logreg_model = linear_model.LogisticRegression(penalty='l2', solver = 'lbfgs',
                                                max_iter = 1000, C=C)
    # fit and score on training
    logreg_model.fit(training_Xi, training_Yi)
    logreg_model.score(training_Xi,training_Yi)

    # score on the validation data
    accuracy = logreg_model.score(val_Xi, val_Yi)
    scores.append(accuracy)

      
  return np.mean(scores)