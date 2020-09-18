import numpy as np

def solve_triang_sup(A,b):
    n = len(b)
    xcomp = np.zeros(n)

    for i in range(n-1, -1, -1):
        tmp = b[i]
        for j in range(n-1, i, -1):
            tmp -= xcomp[j]*A[i,j]
            
        xcomp[i] = tmp/A[i,i]
    return xcomp

def gram_schmidt(V):
    # given m vectors of length m, returns a orthonormal
    # base obtained by Gram-Schmidt process.
    # it will only return the orthonormal vectors quantity given
    # by the shape of U: if rows>cols it will return cols vectors of length rows,
    # otherwise if rows<cols, it will return rows vectors of lenght rows

    # get the matrix shape:
    #   m: number of rows
    #   n: number of cols
    m = len(A)
    n = len(A[0])

    # get the number of vectors to be computed: the min between cols(n) and rows(m)
    k = min([m,n])
    # initialize Q with zeros
    Q = np.zeros((m,k))
    
    # set Q first column to normalized A[:,0]
    Q[:,0] = V[:,0] / np.linalg.norm(V[:,0])


    # iterate from 1 to k-1, to generate the last k-1 ortonormal vectors
    for i in range(1,k):
        Q[:,i] = V[:,i]
        # for the current element, substract the projection over the previous elements
        for j in range(0,i):
            Q[:,i] = Q[:,i] - proj(Q[:,i], Q[:,j])
        # normalize the current vector
        Q[:,i] = Q[:,i] / np.linalg.norm(Q[:,i])

    return Q

def proj(v,u):
    # given two vectors with the same length, returns the 
    # projection of v over u.
    return np.dot(v,u)/np.dot(u,u)*u

def leastsq(A,b):

    # get shape of A.   
    #   m: rows
    #   n: columns
    m = len(A)
    if m == 0:
        return np.array([])
    n = len(A[0])
    
    # if data is empty, return empty result
    if m == 0 and n == 0:
        return np.array([])

    # TODO: what if row count is 1?
    if m == 1:
        return np.array([0])

    # first run GM:
    Q1 = gram_schmidt(A)

    # compute R1
    R1 = np.matmul(np.transpose(Q1), A)

    #get the number of vectors returned by the ortho-normalization process
    Q = Q1
    k = len(Q1[0])

    # case 2:
    if m > n:
        for i in range(m-n):
            # get the number of columns from Q
            k = len(Q[0])
            # generate the random vector
            r = np.random.rand(m,k+1)
            
            for j in range(0,k):
                # substract the projection over the previous vectors
                r[:,k] = r[:,k] - proj(r[:,k],Q[:,j])
            # normalize the new vector
            r[:,k] = r[:,k] / np.linalg.norm(r[:,k])
            # append new vecotr to Q
            r[:,:-1] = Q
            # redefine Q
            Q = np.array(r)

    Q2 = Q[:,len(Q1[0]):]
    return solve_triang_sup(R1,np.transpose(Q1).dot(b))

def test():
    #funciton to test leastsq()

    # TODO: fill cases data as follows
    # To define numpy matrix A: 
    # A = np.array( [ [1,2], [3,4], [5,6] ] ) = |1 2|
    #                                           |3 4|
    #                                           |5 6|
    #
    # A = np. array([ [1, 2, 3], [4, 5, 6] ]) = |1 2 3|
    #                                           |4 5 6|
    #
    # To define numpy array b:
    # Random: np.random.rand(n)
    # Defined: np.array([1, 2, 3]) = |1 2 3|
    # Zero: np.zeros(n) 
    cases = [
        # empty data
        {"A":np.array([]), "b":np.array([])}, #1
        # m < n cases
        {"A":np.array([]), "b":np.array([])}, #2
        {"A":np.array([]), "b":np.array([])}, #3
        {"A":np.array([]), "b":np.array([])}, #4
        {"A":np.array([]), "b":np.array([])}, #5
        # m = n cases
        {"A":np.array([]), "b":np.array([])}, #6
        {"A":np.array([]), "b":np.array([])}, #7
        {"A":np.array([]), "b":np.array([])}, #8
        {"A":np.array([]), "b":np.array([])}, #9
        # m > n cases
        {"A":np.array([]), "b":np.array([])}, #10
        {"A":np.array([]), "b":np.array([])}, #11
        {"A":np.array([]), "b":np.array([])}, #12
        {"A":np.array([]), "b":np.array([])}  #13
    ]

    print('Test started')
    i = 1
    for case in cases:
        print(10*'-' + 'Test {}'.format(i) + 10*'-')
        print('')
        A = case["A"]
        b = case["b"]
        x = leastsq(A,b)
        print('A:')
        print(A)
        print('b:')
        print(b)
        print('x:')
        print(x)
        i = i+1
        print('')

    print(27*'-')
    print('Test completed')


test()