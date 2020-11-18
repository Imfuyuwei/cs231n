from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                
                # Subtract X[i].T from the y[i]-th column of dW
                dW[:, y[i]] -= X[i].T
                
                # Add X[i].T to the j-th column of dW if j != y[i]
                dW[:, j] += X[i].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Want dW to be an average, divide it by num_train
    dW /= num_train

    # Add regularization to the gradient, which is the derivative of loss with respect to W
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    
    # Diagonal elements are correct scores. Get the correct scores for every class, it is a (N,) matrix.
    correct_scores = scores[np.arange(num_train), y]
    
    # Clone the correct_scores column num_classes times to get a (N, C) matrix.
    correct_scores_matrix = np.array([correct_scores,] * num_classes).transpose()
    
    # Use scores to subtract correct scores.
    margins = np.maximum(0, scores - correct_scores_matrix + 1)
    
    # Diagonal elements in margins should be 0.
    margins[np.arange(num_train), y] = 0
    
    loss = np.sum(margins)
    loss /= num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Create a margins_for_dW which has the same shape as margins
    margins_for_dW = np.zeros(margins.shape)
    
    # If there is a positive value in margins, the corresponding element in margins_for_dW is 1.
    margins_for_dW[margins > 0] = 1
    
    # Count the number of positive values in margins
    sum_Of_row = np.sum(margins_for_dW, axis = 1)
    
    # The diagonal elements in margins_for_dW should be the opposite value of sum_Of_row.
    margins_for_dW[np.arange(num_train), y] = -sum_Of_row
    
    # Use X.T multiply sum_Of_row
    dW = X.T.dot(margins_for_dW)/num_train + 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
