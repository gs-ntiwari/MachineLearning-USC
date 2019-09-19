"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    if w is None and y is None:
        return None
    
    if w is None:
        y.reshape((len(y),1))
        return (np.square(np.subtract(X, np.array(y, np.float64)))).mean()
    
    predictions=np.dot(X,w)
    err = (np.square(np.subtract(predictions, np.array(y, np.float64)))).mean(axis=None)
    #print(err, w)
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
      Compute the weight parameter given X and y.
      Inputs:
      - X: A numpy array of shape (num_samples, D) containing feature.
      - y: A numpy array of shape (num_samples, ) containing label
      Returns:
      - w: a numpy array of shape (D, )
      w∗ = (XtX)−1Xty
      """

      #####################################################
      #	TODO 2: Fill in your code here #
      #####################################################
    XtX = np.dot(np.transpose(X),X)
    #print(XtX.shape)
    inverseXtX= np.linalg.inv(XtX)
    inverseXtXXt=np.dot(inverseXtX,np.transpose(X))
    #print(inverseXtXXt.shape, y.shape)
    w = np.dot(inverseXtXXt, y)
    return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    w = None
    newX=correct_eigen_values(np.dot(X,np.transpose(X)))
    #print(newX.shape)
    inversedX=np.linalg.inv(newX)
    temp=np.dot(inversedX, X)
    w=np.dot(np.transpose(temp), y)
    return w

def correct_eigen_values(X):
    newX=X
    eigvals=np.absolute(np.linalg.eigvals(X))
    min_val =np.min(eigvals)
    #print('min_val', min_val)
    if min_val<0.00001:
        X=np.add(X, np.identity(len(X))*0.1)
        newX=correct_eigen_values(X)
    #print(newX.shape)
    return newX
    


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    #print('regularized_linear_regression',X.shape, y.shape, lambd)
    XtX = np.dot(np.transpose(X),X)
    #print(XtX.shape)
    XtX= np.add(XtX,np.identity(len(XtX[0]))*lambd)
    #print(XtX.shape)
    inverseXtX= np.linalg.inv(XtX)
    #print(inverseXtX.shape)
    inverseXtXXt=np.dot(inverseXtX,np.transpose(X))
    #print(inverseXtXXt.shape, y.shape)
    w = np.dot(inverseXtXXt, y)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    currentLambda =round(10**(-19), 19)
    power=-19
    last_lambda=round(10**(20),0)
    min_error=float("inf")
    bestlambda = None
    while currentLambda<=last_lambda:
        #print("currentLambda",currentLambda)
        w_train = regularized_linear_regression(Xtrain, ytrain, currentLambda)
        mse_val = mean_square_error(w_train, Xval, yval)
        #print("mse_val",mse_val)
        if min_error>mse_val:
            min_error=mse_val
            bestlambda=currentLambda
        currentLambda*=10
        power+=1
        currentLambda=round(currentLambda,abs(min(0, power)))
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################		
    temp=power-1
    currentX=X
    final_X=X
    while temp!=0:
        currentX=np.multiply(currentX, X)
        #print(currentX.shape)
        final_X=np.append(final_X, values=currentX, axis=1)
        #print(currentX.shape)
        temp-=1
    #print(X.shape,final_X.shape)
    return final_X


