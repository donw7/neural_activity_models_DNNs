import numpy as np
from sklearn import linear_model
import decoders


# get k-fold splits
def get_test_train_splits(data, decisions, n_folds=5):
  """
  Returns a tuple of matched train sets and validation sets,
  rotating through the data.
  Used by multiple function wrappers as below.  
  """
  fold_size = len(data)//n_folds
  
  training_X = [np.roll(data,fold_size*i, axis=0)[fold_size:] for i in range(n_folds)]
  val_X = [np.roll(data,fold_size*i, axis=0)[:fold_size] for i in range(n_folds)]
  
  training_Y = [np.roll(decisions,fold_size*i, axis=0)[fold_size:] for i in range(n_folds)]
  val_Y = [np.roll(decisions,fold_size*i, axis=0)[:fold_size] for i in range(n_folds)]
  
  return (training_X, training_Y), (val_X, val_Y)

# -----------------------------------------------------------------------------
# kfold val function for hardcoded logistic regression
# with customized C regularization parameter
def get_kfold_validation_score(data, decisions, C):
  '''
  gets mean validation accuracy with C reg without plotting or other shenanigans

  data: expects numpy array of shape (n_trials, n_units, n_bins)
  decisions: expects numpy array of shape (n_trials, 1)
    can be generalized to most categorical situations  
  
  Output: returns mean validation accuracy of all kfold accuracies
  '''

  # use func from above to get splits
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

# -----------------------------------------------------------------------------
# kfold training wrapper with helper function
def compute_accuracy(val_spikes, val_choices, model):
  predictions = model.predict(val_spikes)
  accuracy = np.sum(predictions == val_choices) / len(predictions)
  return accuracy

def kfold_train_wrapper(model_name, data, decisions,
                        n_folds, units, dropout, num_epochs, verbose):
  '''
  wrapper function to train and score on k-fold

  Inputs: 
  ----------
  model_name: calls str model_name as class in decoders module
  data: expects numpy array of shape (n_trials, n_units, n_bins)
  decisions: expects numpy array of shape (n_trials, 1)
    can be generalized to most categorical situations  
  
  Parameters: 
  ----------
  n_folds: number of folds to use for k-fold cross validation
  units: Number of hidden units in each layer
  dropout: Proportion of units that get dropped out
  num_epochs: Number of epochs used for training
  verbose: binary, whether to show progress of the fit after each epoch
  
  Output: returns list of all kfold accuracies
  '''

  all_train_accuracy = []
  all_val_accuracy = []

  # use func from above to get splits
  (training_X, training_Y), (validation_X, validation_Y) = get_test_train_splits(data, decisions)
    
  for fold in range(n_folds):
      print(f"Fold {fold} of {n_folds}")
      train_Xi = training_X[fold]
      train_Yi = training_Y[fold]
      val_Xi = validation_X[fold]
      val_Yi = validation_Y[fold]
      
      method_to_call = getattr(decoders, model_name) # as class in separate module
      model = method_to_call(units=units, dropout=dropout, num_epochs=num_epochs, verbose=verbose)

      model.fit(train_Xi, train_Yi)
      train_accuracy = compute_accuracy(train_Xi, train_Yi, model)
      all_train_accuracy.append(train_accuracy)

      # score on the validation data
      val_accuracy = compute_accuracy(val_Xi, val_Yi, model)
      all_val_accuracy.append(val_accuracy)

      print("training fold accuracy: {}".format(train_accuracy))
      print("validation foldaccuracy: {}".format(val_accuracy))

  return all_train_accuracy, all_val_accuracy