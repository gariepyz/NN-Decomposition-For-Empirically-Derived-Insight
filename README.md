# Novel Neural-Network-Decomposition-Algorithm
Code used for the publication "Machine-Learning-Driven High-Entropy Alloy Catalyst Discovery to Circumvent the Scaling Relation for CO2 Reduction Reaction"
ACS Catal. 2022, 12, 24, 14864–14871 (https://doi.org/10.1021/acscatal.2c03675)

There are 2 notebooks and accompanying files/datasets in this repository:

1: 'Neural_Net_Decomposition_Method'is a notebook outlining how to train a basic NN model with TF, extract internal weights, and perform a NN decomposition outlined in the publication. Required csv is 'All_data.csv'

2: 'Sample_data_visualization' is a notebook providing basic data visualization for the figures provided in the publication. Required csvs are 'All_data.csv','site_infs.csv'

3: 'helpers.py' is a python file with a classes for feature embedding and the NN decomposition.

Required Python Packages to run the code:
pandas,
numpy,
matplotlib,
seaborn,
math,
tensorflow,
sklearn
