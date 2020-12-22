# CreditFraud
Python notebook for Credit Card fraud detection

This notebook implements SMOTE sampling to handle data imbalances since the number of fraud instances are far fewer than non-fraud ones.
SMOTE creates synthetic data points in order to balance the number of data points for each class. (The creation happens during cross validation)

This helps oversample our training data during cross-validation and we use a confusion matrix & other metrics like precision-recall score and f1 scores as a metric instead of a misleading accuracy score.

This notebook has a logistic regression mehtod and a neural network implementation.
Further details are provided in the notebook.
