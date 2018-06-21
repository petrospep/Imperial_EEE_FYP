import numpy as np
def burst_error_array(p1, p2,dim_z):
    # Markov State Error
    pp = [p1, 1 - p2]
    J = 100
    state_z = np.zeros((J, dim_z))

    for j in range(J):
        state_z[j, 0] = np.random.choice(2, 1, p=[0.9, 0.1])
        for i in range(1, dim_z):
            p = pp[state_z[j, i - 1].astype(np.int32)]
            state_z[j, i - 1] = np.random.choice(2, 1, p=[p, 1 - p])

    return state_z


#Save mean,std pair
def save_statistics(mu,sigma,deterministic):

    Distributions=np.zeros(mu.shape,dtype=np.dtype('<U16'))

    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            miou=mu[i,j]
            zigma=sigma[i,j]
            if deterministic==0:
                temp='(%1.3f,%1.3f)'%(miou,zigma)
                Distributions[i,j]=temp
            if deterministic == 1:
                temp = '(%1.3f)' % (miou)
                Distributions[i, j] = temp

    return Distributions