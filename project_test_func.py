import numpy as np

#################################################
### Conjunto de funções teste e de amostragem ###
#################################################

def adjust_index(x, lims, spacing):

    if len(x.shape) == 1: 
        dim = x.shape[0]
        aux = np.zeros(x.shape)      
        for i in range(dim):

            aux0 = int((x[i]-lims[0])/spacing)
            aux1 = int((lims[1]-lims[0])/spacing)
            aux2 = int(min([aux0, aux1]))
            
            aux[i] = int(max(aux2, 0))

    else:
        dim = x.shape[1]
        aux = np.zeros(x.shape)       
        for i in range(dim):

            aux0 = ((x[:, i] - lims[0])/spacing).astype(int)
            aux1 = ((lims[1]-lims[0])/spacing).astype(int)
            aux2 = np.ones(x[:, i].shape) * aux1
            aux3 = np.r_["0,2", aux0, aux2]
            aux4 = np.min(aux3, axis=0).astype(int)
            aux5 = np.zeros(aux4.shape)
            aux6 = np.r_["0,2", aux4, aux5]
            
            aux[:, i] = np.max(aux6, axis=0).astype(int)


    return aux.astype(int)

def adjust_pos(x, lims, spacing):

    if len(x.shape) == 1: 
        dim = x.shape[0]
        aux = np.zeros(x.shape)      
        for i in range(dim):

            aux0 = int((x[i]-lims[0])/spacing)
            aux1 = int((lims[1]-lims[0])/spacing)
            aux2 = int(min([aux0, aux1]))
            aux3 = int(max(aux2, 0))

            aux[i] = aux3*spacing + lims[0]
    else:
        dim = x.shape[1]
        aux = np.zeros(x.shape)       
        for i in range(dim):

            aux0 = ((x[:, i] - lims[0])/spacing).astype(int)
            aux1 = ((lims[1]-lims[0])/spacing).astype(int)
            aux2 = np.ones(x[:, i].shape) * aux1
            aux3 = np.r_["0,2", aux0, aux2]
            aux4 = np.min(aux3, axis=0).astype(int)
            aux5 = np.zeros(aux4.shape)
            aux6 = np.r_["0,2", aux4, aux5]
            aux7 = np.max(aux6, axis=0).astype(int)

            aux[:, i] = aux7*spacing + lims[0]*np.ones(aux7.shape)

    return aux

def sample_func(fun, x, lims, spacing):

    adjust_x = adjust_pos(x, lims, spacing)
    sample = fun(adjust_x, lims, spacing)

    return sample

def mixer(arr, samples=1, seed=0):

    if samples > 1:
        mix = []
    aux = arr ** 1
    dim = aux.shape

    for i in range(samples):
        aux0 = aux.flatten()
        aux1 = np.random.RandomState(seed).permutation(aux0)
        aux = aux1.reshape(dim)
        if samples > 1:
            mix.append(aux.copy())

    if samples > 1:
        return mix
    else:
        return aux

def matrix_equiv(arr):

    rots = [arr, np.rot90(arr, 1), np.rot90(arr, 2), np.rot90(arr, 3)]
    reflex = [np.fliplr(matrix) for matrix in rots] +  [np.flipud(matrix) for matrix in rots]
    eqs = {tuple(matrix.ravel()) for matrix in rots + reflex}
    
    return eqs

def construct(fun, lims, spacing, samples=1):

    num_points = int((lims[1]-lims[0])/spacing)
    x, y = np.linspace(lims[0], lims[1], num_points), np.linspace(lims[0], lims[1], num_points)
    x, y = np.meshgrid(x, y)
    aux_matrix = fun(x, y)
    matrix = mixer(aux_matrix, samples)

    if samples > 1:
        final_matrix = []
        uniq_matrix = set()
        for i in range(samples):
            eqs = matrix_equiv(matrix[i])
            if not uniq_matrix.intersection(eqs):
                uniq_matrix.add(tuple(matrix[i].ravel()))
                final_matrix.append(matrix[i])

        return final_matrix
    
    else:
        return matrix


def rosenbr(x, lims, spacing):

    if len(x.shape) == 1:
        aux0 = (1 - x[0])
        aux1 = (x[1] - x[0]**2)
        aux2 = aux0**2
        aux3 = 100 * aux1**2
    else:
        aux0 = (1 - x[:, 0])
        aux1 = (x[:, 1] - x[:, 0]**2)
        aux2 = aux0**2
        aux3 = 100 * aux1**2
    
    return aux2 + aux3

def rosenbr_2D(x, y):

    aux0 = (1 - x)
    aux1 = (y - x**2)
    aux2 = aux0**2
    aux3 = 100 * aux1**2
    
    return aux2 + aux3

def rosenbr_n(x, lims, spacing):

    if len(x.shape) == 1: 
        dim = x.shape[0]
        aux = 0.0       
        for i in range(dim-1):

            aux0 = ( - x[i] + 1)
            aux1 = (x[i+1] - x[i]**2)
            aux2 = aux0**2
            aux3 = 100 * aux1**2
            
            aux += aux2 + aux3
    else:
        dim = x.shape[1]
        aux = np.zeros(x.shape[0])       
        for i in range(dim-1):

            aux0 = ( - x[:, i] + 1)
            aux1 = (x[:, i+1] - x[:, i]**2)
            aux2 = aux0**2
            aux3 = 100 * aux1**2

            aux += aux2 + aux3
    return aux

def rosenbr_n_sample(x, lims, spacing):
    
    sample = sample_func(rosenbr_n, x, lims, spacing)

    return sample


def ackley(x, lims, spacing):

    if len(x.shape) == 1:
        aux0 = 0.5 * (x[0]**2 + x[1]**2)
        aux1 = - 0.2 * np.sqrt(aux0)
        aux2 = - 20 * np.exp(aux1)

        aux3 = np.cos( 2 * np.pi * x[0]) + np.cos( 2 * np.pi * x[1])
        aux4 = - np.exp(0.5 * aux3)

        aux5 = np.e + 20
    else:
        aux0 = 0.5 * (x[:, 0]**2 + x[:, 1]**2)
        aux1 = - 0.2 * np.sqrt(aux0)
        aux2 = - 20 * np.exp(aux1)

        aux3 = np.cos( 2 * np.pi * x[:, 0]) + np.cos( 2 * np.pi * x[:, 1])
        aux4 = - np.exp(0.5 * aux3)

        aux5 = np.e + 20

    return aux2 + aux4 + aux5

def ackley_2D(x, y):

    aux0 = 0.5 * (x**2 + y**2)
    aux1 = - 0.2 * np.sqrt(aux0)
    aux2 = - 20 * np.exp(aux1)

    aux3 = np.cos( 2 * np.pi * x) + np.cos( 2 * np.pi * y)
    aux4 = - np.exp(0.5 * aux3)

    aux5 = np.e + 20
    
    return aux2 + aux4 + aux5

def ackley_sample(x, lims, spacing):

    sample = sample_func(ackley, x, lims, spacing)

    return sample

def ackley_n(x, lims, spacing):
    
    a, b, c = 20, 0.2, 2*np.pi

    if len(x.shape) == 1: 
        dim = x.shape[0]
        aux = 0.0
        aux0 = 0.0
        aux1 = 0.0
        for i in range(dim):
            aux0 += x[i]**2
            aux1 += np.cos(c*x[i])
        
        aux2 = -b*np.sqrt((1/dim)*aux0)
        aux3 = -a*np.exp(aux2)

        aux4 = -np.exp((1/dim)*aux1)
        aux = aux3 + aux4 + a + np.exp(1)

    else:
        dim = x.shape[1]
        aux = np.zeros(x.shape[0]) 
        aux0 = np.zeros(x.shape[0])
        aux1 = np.zeros(x.shape[0])
        for i in range(dim):
            aux0 += x[:, i]**2
            aux1 += np.cos(c*x[:, i])
        
        aux2 = -b*np.sqrt((1/dim)*aux0)
        aux3 = -a*np.exp(aux2)

        aux4 = -np.exp((1/dim)*aux1)
        aux = aux3 + aux4 + a + np.exp(1)

    return aux


def easom(x, lims, spacing):

    if len(x.shape) == 1:
        aux0 = - np.cos(x[0])*np.cos(x[1])
        aux1 = (x[0] - np.pi)**2 + (x[1] - np.pi)**2
        aux2 = np.exp(-aux1)
        aux3 = aux0 * aux2
    else:
        aux0 = - np.cos(x[:, 0])*np.cos(x[:, 1])
        aux1 = (x[:, 0] - np.pi)**2 + (x[:, 1] - np.pi)**2
        aux2 = np.exp(-aux1)
        aux3 = aux0 * aux2

    return aux3

def easom_2D(x, y):

    aux0 = - np.cos(x)*np.cos(y)
    aux1 = (x - np.pi)**2 + (y - np.pi)**2
    aux2 = np.exp(-aux1)
    aux3 = aux0 * aux2

    return aux3

def easom_sample(x, lims, spacing):

    sample = sample_func(easom, x, lims, spacing)

    return sample


def rastrigin(x, lims, spacing):

    if len(x.shape) == 1:
        aux0 = x[0]**2 - 10*np.cos(2*np.pi * x[0])
        aux1 = x[1]**2 - 10*np.cos(2*np.pi * x[1])
        aux2 = aux0 + aux1
    else:
        aux0 = x[:, 0]**2 - 10*np.cos(2*np.pi * x[:, 0])
        aux1 = x[:, 1]**2 - 10*np.cos(2*np.pi * x[:, 1])
        aux2 = aux0 + aux1
    
    return aux2 + 20

def rastrigin_2D(x, y):

    aux0 = x**2 - 10*np.cos(2*np.pi * x)
    aux1 = y**2 - 10*np.cos(2*np.pi * y)
    aux2 = aux0 + aux1

    return aux2 + 20

def rastrigin_sample(x, lims, spacing):

    sample = sample_func(rastrigin, x, lims, spacing)

    return sample

def rastrigin_n(x, lims, spacing):

    if len(x.shape) == 1: 
        dim = x.shape[0]
        aux = 0.0
        aux0 = 0.0 
        for i in range(dim):
            aux0 += x[i]**2 - 10*np.cos(x[i]*2*np.pi)
        aux = 10*dim + aux0

    else:
        dim = x.shape[1]
        aux = np.zeros(x.shape[0]) 
        aux0 = np.zeros(x.shape[0])
        for i in range(dim):
            aux0 += x[:, i]**2 - 10*np.cos(x[:, i]*2*np.pi)
        aux = 10*dim + aux0

    return aux


def eggholder(x, lims, spacing):

    if len(x.shape) == 1:
        aux0 = np.sqrt(np.abs(0.5 * x[0]  + x[1] + 47.0))
        aux1 = - (47.0 + x[1]) * np.sin(aux0) 

        aux2 = np.sqrt(np.abs(x[0] - x[1] - 47.0))
        aux3 = - x[0] * np.sin(aux2)
    else:
        aux0 = np.sqrt(np.abs(0.5 * x[:, 0]  + x[:, 1] + 47.0))
        aux1 = - (47.0 + x[:, 1]) * np.sin(aux0) 

        aux2 = np.sqrt(np.abs(x[:, 0] - x[:, 1] - 47.0))
        aux3 = - x[:, 0] * np.sin(aux2)
    
    return aux1 + aux3

def eggholder_2D(x, y):

    aux0 = np.sqrt(np.abs(0.5 * x  + y + 47.0))
    aux1 = - (47.0 + y) * np.sin(aux0) 

    aux2 = np.sqrt(np.abs(x - y - 47.0))
    aux3 = - x * np.sin(aux2)

    return aux1 + aux3

def eggholder_sample(x, lims, spacing):

    sample = sample_func(eggholder, x, lims, spacing)

    return sample


def drop_wave(x, lims, spacing):

    aux0 = 1.0 + np.cos(12*np.sqrt(x[0]**2 + x[1]**2))
    aux1 = 2.0 + 0.5*(x[0]**2 + x[1]**2)

    return - aux0/aux1

def drop_wave_sample(x, lims, spacing):

    sample = sample_func(drop_wave, x, lims, spacing)

    return sample


def styblinski_tang(x, lims, spacing):

    if len(x.shape) == 1: 
        dim = x.shape[0]
        aux = 0.0       
        for i in range(dim):
            aux0 = x[i]**4 - 16*x[i]**2 + 5*x[i]
            aux += aux0
    else:
        dim = x.shape[1]
        aux = np.zeros(x.shape[0]) 
        for i in range(dim):
            aux0 = x[:, i]**4 - 16*x[:, i]**2 + 5*x[:, i]
            aux += aux0

    return 0.5*aux

def styblinski_tang_sample(x, lims, spacing):

    sample = sample_func(styblinski_tang, x, lims, spacing)

    return sample


def michalewicz(x, lims, spacing):

    if len(x.shape) == 1: 
        dim = x.shape[0]
        aux = 0.0       
        for i in range(dim):
            aux0 = np.sin(x[i])
            aux1 = ((i+1)*(x[i]**2))/np.pi
            aux2 = (np.sin(aux1))**(20)
            aux += aux0*aux2
    else:
        dim = x.shape[1]
        aux = np.zeros(x.shape[0]) 
        for i in range(dim):
            aux0 = np.sin(x[:, i])
            aux1 = ((i+1)*(x[:, i]**2))/np.pi
            aux2 = (np.sin(aux1))**(20)
            aux += aux0*aux2

    return - aux

def michalewicz_sample(x, lims, spacing):

    sample = sample_func(michalewicz, x, lims, spacing)

    return sample


def perm(x, lims, spacing):

    if len(x.shape) == 1: 
        dim = x.shape[0]
        aux = 0.0       
        for i in range(1, dim + 1):
            aux0 = 0.0
            for j in range(1, dim + 1):
                aux1 = (j ** i + 0.5)
                aux2 = ((x[j - 1] / j) ** i - 1)
                aux0 += aux1* aux2
            aux += aux0 ** 2

    else:
        dim = x.shape[1]
        aux = np.zeros(x.shape[0]) 
        for i in range(1, dim + 1):
            aux0 = np.zeros(x.shape[0])
            for j in range(1, dim + 1):
                aux1 = (j ** i + 0.5)
                aux2 = ((x[:, j - 1] / j) ** i - 1)
                aux0 += aux1* aux2
            aux += aux0 ** 2

    return aux


def griewank(x, lims, spacing):

    if len(x.shape) == 1: 
        dim = x.shape[0]
        aux = 0.0
        aux0 = 0.0
        aux1 = 1.0    
        for i in range(1, dim+1):
            aux0 += (x[i-1]**2)/4000
            aux1 *= np.cos(x[i-1]/np.sqrt(i))
        aux = aux0 - aux1 + 1

    else:
        dim = x.shape[1]
        aux = np.zeros(x.shape[0]) 
        aux0 = np.zeros(x.shape[0])
        aux1 = np.ones(x.shape[0])    
        for i in range(1, dim+1):
            aux0 += (x[:, i-1]**2)/4000
            aux1 *= np.cos(x[:, i-1]/np.sqrt(i))
        aux = aux0 - aux1 + 1

    return aux


def schwefel(x, lims, spacing):

    if len(x.shape) == 1: 
        dim = x.shape[0]
        aux = 0.0
        aux0 = 0.0 
        for i in range(dim):
            aux0 += x[i]*np.sin(np.sqrt(np.abs(x[i])))
        aux = 418.9829*dim - aux0

    else:
        dim = x.shape[1]
        aux = np.zeros(x.shape[0]) 
        aux0 = np.zeros(x.shape[0])
        for i in range(dim):
            aux0 += x[:, i]*np.sin(np.sqrt(np.abs(x[:, i])))
        aux = 418.9829*dim - aux0

    return aux
