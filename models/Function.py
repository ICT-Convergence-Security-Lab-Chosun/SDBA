import copy
import math
import torch
from torch import nn
import numpy as np

#####################
##### Distances #####
#####################


## Cosine Similarity
def CosSim(w):
    # score is the result for this function
    score = np.array([[0. for i in range(len(w))] for j in range(len(w))]) ## n by n matrix
    w_avg = copy.deepcopy(w[0]) # To use keys in dict in w
    norm_total = [0. for i in range(len(w))] # Denominator for CosSim

    tmp = copy.deepcopy(w) # To compute CosSim

    ### Make flat matrix for all w[i] ###
    flat_mat = [np.array([], dtype=np.float32) for i in range(len(w))]

    for k in w_avg.keys():
        for i in range(len(w)):
            A = tmp[i][k].cpu().numpy()
            flat_A = A.flatten()
            flat_mat[i] = np.concatenate((flat_mat[i], flat_A), axis=0)

    print(flat_mat)

    #### CosSim(A, B) ####
    for i in range(len(w)):
        for j in range(len(w)):
            if j < i or j == i: ## j < i is included because score is sym matrix
                continue
            else:
                norm_A = np.linalg.norm(flat_mat[i], ord=2)
                norm_B = np.linalg.norm(flat_mat[j], ord=2)
                numerator = np.dot(flat_mat[i], flat_mat[j])
                score[i][j] = numerator / (norm_A * norm_B)


    # Make symmetric matrix
    # Because we cannot compute for j < i for better computational complexity(De-duplicating for computing)
    score += score.T - np.diag(score.diagonal())
    #norm_total = np.sqrt(norm_total)

    for i in range(len(w)):
        for j in range(len(w)):
            if i == j:
                score[i][j] = 1.
            #else:
            #    score[i][j] = score[i][j] / (norm_total[i] * norm_total[j])

    print("score check for CosSim")
    print(np.shape(score))
    print(score)
    print()

    return score


def CosSim_with_key(w):
    # score is the result for this function
    score = {}
    norm_total = {}
    w_avg = copy.deepcopy(w[0]) # To use keys in dict in w
    for k in w_avg.keys():
        score[k] = np.array([[0. for i in range(len(w))] for j in range(len(w))]) ## n by n matrix
        norm_total[k] = [0. for i in range(len(w))] # Denominator for CosSim
    tmp_A = {}
    tmp = copy.deepcopy(w) # To compute CosSim
    for k in w_avg.keys(): ## for each key
        for i in range(len(w)): ## for each local models
            ## Compute Denominator ##
            A = tmp[i][k].cpu().numpy()
            tmp_A[k] = np.linalg.norm(A, ord=None) ## l2-norm
            #tmp_A = tmp_A ** 2
            norm_total[k][i] += tmp_A[k]
            flat_A = A.flatten()

            ## Compute numerator ##
            for j in range(len(w)):
                if j < i or j == i: ## because score is symmetric matrix
                    continue
                else:
                    B = tmp[j][k].cpu().numpy()
                    flat_B = B.flatten()
                    score[k][i][j] += np.dot(flat_A, flat_B) # Insert each numerator value into score matrix first


    # Make symmetric matrix
    # Because we cannot compute for j < i for better computational complexity(De-duplicating for computing)
    for k in w_avg.keys():
        score[k] += score[k].T - np.diag(score[k].diagonal())
    #norm_total = np.sqrt(norm_total)
    for k in w_avg.keys():
        for i in range(len(w)):
            for j in range(len(w)):
                if i == j:
                    score[k][i][j] = 1.
                else:
                    score[k][i][j] = score[k][i][j] / (norm_total[k][i] * norm_total[k][j])

    #print(score)

    return score

