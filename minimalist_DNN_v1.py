"""

Minimalist DNN

"""

import numpy as np

# define cost
objective = lambda Y, Y_hat : 0.5*(Y - Y_hat)**2
objective_grad_dY_hat = lambda Y, Y_hat : Y_hat-Y

# feed forward
def feed_forward( X, Ws ):
    As = [X.T]
    for W in Ws:
        As += [W @ As[-1]]
    # A[0], ..., A[-1] = Y_hat
    return As

# predict class labels
def predict( X, Ws ):
    As = feed_forward( X, Ws ) # A[0], ..., A[-1] = Y_hat
    Y_pred = np.argmax(As[-1], axis=0)
    return Y_pred

def mse( Y, Y_hat ):
    cost = np.mean((Y_hat-Y)**2)
    return cost

from sklearn.metrics import accuracy_score
def accuracy( Y, Y_hat ):
    Y_pred = np.argmax(Y_hat, axis=0)
    Y_true = np.argmax(Y, axis=0)
    return accuracy_score(Y_true, Y_pred)

# back propigate
def back_propigate( X, Y, Ws, As ):

    # get Y_hat
    Y_hat = As[-1]


#     print( objective(Y, Y_hat) )
    
#     print("\nY_hat:")
#     print(Y_hat.shape)

#     print("\nX.T:")
#     print(X.T.shape)

#     print("\nAs:")
#     for A in As:
#         print(A.shape)

#     print("\nWs:")
#     for W in Ws:
#         print(W.shape)


    # initilize sensitivies list
    Vs = [0]*(L+1)

    #for A in As:
    #    Vs.append(np.zeros((1,1)))

    # calculate final sensitivity
    V_final = objective_grad_dY_hat(Y, Y_hat) # Y_hat-Y # add  * dphi(Y_hat)
    Vs[-1] = V_final #objective_grad_dY_hat(Y, Y_hat) # Y_hat-Y # add  * dphi(Y_hat)
    

#     # ERROR
#     # calculate second to last sensitivity
#     Vs[-2] = Ws[-1].T @ Vs[-1] # no bias terms to remove

#     # calculate remaining sensitivities (must remove biases)
#     for i in range(L-2,-1,-1):
#         Vs[i] = Ws[i+1].T @ Vs[i+1] 
#     # ERROR
    
    # calculate second to last sensitivity
    Vs[-2] = Vs[-1] #Ws[-1].T @ Vs[-1] # no bias terms to remove

    # calculate remaining sensitivities (must remove biases)
    for i in range(L-2,-1,-1):
        Vs[i] = Ws[i+1].T @ Vs[i+1] 

#     # display sensitivities
#     print("\nVs:")
#     for V in Vs:
#         print(V.shape)

    # initilize W_grads
    W_grads = [0]*L
    # calculate final W gradient
    W_grad_n = Vs[-1] @ As[-2].T # no bias to remove As[-2] is A before Y_hat
    W_grads[-1] = W_grad_n

    # caclulate remaing gradients
    #for i in range(L-1,0,-1):
    for i in range(L-1):
        W_grad = Vs[i] @ As[i].T
        W_grads[i] = W_grad

    #[TODO] regularize weights that are not bias terms

#     print("\nW_grads:")
#     for grad in W_grads:
#         print(grad.shape)

    # update Ws
    eta = 0.2
    Updated_Ws = []
    for W, W_grad in zip(Ws, W_grads):
        # update equation
        W_updated = W - eta*W_grad
        Updated_Ws += [ W_updated ]

    # return updated Ws
    return Updated_Ws

# train network
def train(X, Y, Ws, epochs):
    
    for i in range(epochs):
        
        As = feed_forward(X, Ws)
        Ws = back_propigate(X, Y, Ws, As)
        print_interval = 20
        if i % print_interval == 0:
            As = feed_forward( X, Ws )
            Y_hat = As[-1]
            print(accuracy(Y, Y_hat))

# multiply list of matrices W[0] @ W[1] @ ... @ w[-1]
def mm_list( m_list ):
    R = m_list[0]
    for M in m_list[1:]:
        R = R @ M
    return R

# take transpose of all matrices in list and return list
mT_list = lambda m_list : [ M.T for M in m_list ]

#
# testing
#


#
# create random test data
#

# create random input samples and their classes
n_samples, n_features, n_classes = 7, 15, 5

# create inputs
X = np.random.rand(n_samples, n_features) # Inputs

# create thier classes
Y = np.eye(n_samples, n_classes).T
np.random.shuffle(Y.T)

# get random matrices given shapes
n_hidden = 5

# W[0], ..., W[L-1]
shapes = [ (n_hidden, n_features), (n_hidden, n_hidden), (n_classes, n_hidden) ]
get_Ws = lambda shapes: [ np.random.rand(*s) for s in shapes ]
Ws = get_Ws( shapes )
L = len( Ws )



# train
epochs = 200
   
train(X, Y, Ws, epochs)
