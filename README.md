# Modeling of neural activity using deep neural networks
Deep neural networks are postulated to breakthrough improved performance via learning of high-order features, however limited data and knowledge of _a prior_ structure complicate application to the study of neural activity. Here, I apply SOTA deep neural networks to brain-wide, single neuron recordings enabled by advances in electrophysiology technology to assess considerations for appropriate application and interpretation of results. 

## Decoding decision-making behavior from neural activity

This is a comparison of statistical models vs. deep neural networks (DNNs), their performances across different conditions and with various types of data and hyperparameters.
### Summary excerpted here. see "DNN_decoding.ipynb" for additional details

### Hypotheses:
1. Decision-making behavior can be predicted reasonably well from information encoded in neural activity, even with small datasets
2. DNNs outperform statistical models but are prone to overfitting and require additional considerations, e.g. regularization

### Results & Conclusions summary:
- Complicated decision-making behavior can be predicted reasonably well from information encoded in neural activity, even with small and potentially noisy real-world datasets
- DNNs outperform statistical models (logistic regression accuracy 30-40% whereas DNNs 50-60% vs. 30% expected by chance)
- DNNS are prone to overfitting and require additional considerations, e.g. crossvalidation or regularization
- XGBoost appears to be the best performing DNN (67% accuracy vs. 30% expected by chance)

## Decoding decision-making behavior from neural activity
This is a comparison of statistical models vs. deep neural networks (DNNs), their performances across different conditions and with various types of data and hyperparameters.


