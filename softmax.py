from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)
        
        # In order to avoid numeric instability, make the highest score to be zero
        scores -= np.max(scores)
        
        scores_exp_sum = np.sum(np.exp(scores))
        correct_score_exp = np.exp(scores[y[i]])
        loss += -np.log(correct_score_exp / scores_exp_sum)
        
        # Calculate the gradient of W
        for j in range(num_classes):
            dW[:, j] += - ((j == y[i]) - (np.exp(scores[j])/ scores_exp_sum)) * X[i]
        
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W
        


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    
    # Get the maximum score of every row and clone it multiple times to be a (N, C) matrix
    max_scores = np.amax(scores, axis = 1)
    max_scores_matrix = np.array([max_scores,] * num_classes).transpose()
    scores -= max_scores_matrix
    
    correct_scores = scores[np.arange(num_train), y]
    scores_row_exp_sum = np.sum(np.exp(scores), axis = 1)
    loss = np.sum(-np.log(np.exp(correct_scores) / scores_row_exp_sum))
    
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    
    # Calculate the gradient of W
    margins_for_dW = np.exp(scores) / scores_row_exp_sum.reshape(num_train, 1)
    # The diagonal elements should minus 1 
    margins_for_dW[np.arange(num_train), y] -= 1
    dW = X.T.dot(margins_for_dW)
    
    dW /= num_train
    dW += 2 * reg * W
    
    
    
   
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
