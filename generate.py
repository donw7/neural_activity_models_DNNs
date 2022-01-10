import numpy as np

def neural_data(n_trials, n_neurons = 25, n_bins_per_trial = 50,
                         noise_variance = 1, drift_rate = 0.07,
                         mean_rate = 25):
    """Generates fake neural data of shape (n_trials, n_neurons, n_bins_per_trial
          according to a drift diffusion process with given parameters.
       Also generates decisions, which is 0 or 1 depending on the "animal's decision"
          and is returned as an array of shape (n_trials,)

      Returns: (neural_data, decisions)
    """


    decisions = np.random.binomial(1,.5,size = n_trials)

    neural_recordings = np.zeros((n_trials,n_neurons,n_bins_per_trial))

    for t in range(n_bins_per_trial):
        if t==0:
            neural_recordings[:,:,t] = mean_rate + np.random.randn(n_trials,n_neurons) * noise_variance
        else:
            neural_recordings[:,:,t] = neural_recordings[:,:,t-1] \
                                       + np.reshape(drift_rate*(decisions*2-1),(len(decisions),1)) \
                                       + np.random.randn(n_trials,n_neurons) * noise_variance

    return neural_recordings, decisions