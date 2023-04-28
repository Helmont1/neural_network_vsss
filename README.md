# Neural Network Regression

This project is a implementation of a neural network for regression, trained on a dataset to predict the robot responses in a VSSS soccer game.

## Getting Started
To run this project, you'll need to install the following libraries:

- scikit-learn
- numpy
- pandas
- tensorflow
- jupyterlab

## Usage
The project consists of a single Python script, neural_network.py, which you can run from the command line using the following command:
```python
  python data_analysis.py
```

## Methodology
The dataset used in this project is a collection of VSSS soccer data. The dataset is first preprocessed to remove unnecessary columns, normalize the data, and process categorical data using one-hot encoding. The preprocessed dataset is then split into training, validation, and testing sets. A neural network regression model is created using the TensorFlow library. A randomized search cross-validation technique is used to optimize the hyperparameters of the model. The best model is then trained on the training set and validated on the validation set. The final model is evaluated on the testing set.

## Results
The best model achieved a mean absolute error of 0.23 on the testing set. The model can be used to predict the robot responses in a VSSS soccer game with a medium degree of accuracy.
