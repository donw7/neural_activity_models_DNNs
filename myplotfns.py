import matplotlib.pyplot as plt
import numpy as np

def plot_coefs(model, n_neurons, n_bins_per_trial):
  """Makes a nice plot of the coefficients. fit_model is the model instance after fitting."""


  # get the coefficients of your fit
  coefficients = model.coef_.reshape(n_neurons, n_bins_per_trial)

  # show them
  plt.figure(figsize = (10,5))
  plt.imshow(coefficients, cmap = 'coolwarm', vmin = -np.max(coefficients), 
                                              vmax = np.max(coefficients))

  #make it pretty
  plt.ylabel("Neuron #", fontsize = 14)
  plt.xlabel("Time (ms)", fontsize = 14)
  plt.colorbar(orientation = 'horizontal', shrink = .6, 
               label="Contribution of bin to decision (coeffs)")
  plt.tight_layout()
  plt.show()

  pass