import numpy as np
from project_test_func import *  
from project_FLS import *  


def inside_bounds(x, lims, random_state):

    """
    _________________
    --- Descrição ---
    _________________

    Função para ajustar um ponto no caso de alguma de suas entrdas não estiver contido dentro dos limites. Caso o ponto
    já respeite os limites, nada muda.

    __________________
    --- Argumentos ---
    __________________

    - x : lista representando ponto em análise;

    - lims : lista representando os limites de busca;

    - random_state : uma seed do tipo np.RandomState para gereção de números aleatórios.

    ______________
    --- Saídas ---
    ______________

    - aux_trial : lista representando ponto respeitando oa limites.

    """

    aux_trial = x ** 1
    for cord in range(len(aux_trial)):

        if aux_trial[cord] < lims[0] or aux_trial[cord] > lims[1]:
            aux_trial[cord] = random_state.uniform(low=lims[0], high=lims[1])
        
    return aux_trial

def mutate(index_curr_indiv, curr_population, fun, strat, mut_factor, K_factor, lims, spacing, num_indiv, random_state):
    
    """
    _________________
    --- Descrição ---
    _________________

    Função que aplica a operação de mutação sobre um indivíduo.

    __________________
    --- Argumentos ---
    __________________

    - index_curr_indiv : índice do indivíduo alvo;

    - curr_population : matriz representando população da geração atual;

    - fun : função objetivo;

    - strat : string informando estratégia de mutação utilizada;

    -  mut_factor : coeficiente de mutação F;

    - K_factor : coeficiente de mutação K;

    - lims : lista representando os limites de busca;

    - spacing : espaçamento da malha computacional;

    - num_indiv : número de indivíduos na população;

    - random_state : uma seed do tipo np.RandomState para gereção de números aleatórios.
    ______________
    --- Saídas ---
    ______________

    - mutant : lista representando indivíduo mutante. 

    """

    if strat == "rand/1/bin" :
        
        aux_list = list(range(num_indiv))
        aux_list.pop(index_curr_indiv)

        aux_index = random_state.choice(aux_list, 3, replace=False)
        #aux_index = [1,2,3]
        mutant = curr_population[aux_index[0]] + mut_factor * (curr_population[aux_index[1]] - curr_population[aux_index[2]])
    
    elif strat == "best/1/bin" :

        aux_list = list(range(num_indiv))
        aux_list.pop(index_curr_indiv)

        cost_population = fun(curr_population, lims, spacing)
        index_best = np.argmin(cost_population)

        aux_index = random_state.choice(aux_list, 2, replace=False)
        mutant = curr_population[index_best] + mut_factor * (curr_population[aux_index[0]] - curr_population[aux_index[1]])
    
    elif strat == "best/2/bin" :

        aux_list = list(range(num_indiv))
        aux_list.pop(index_curr_indiv)

        cost_population = fun(curr_population, lims, spacing)
        index_best = np.argmin(cost_population)

        aux_index = random_state.choice(aux_list, 4, replace=False)
        mutant = curr_population[index_best] + mut_factor * (curr_population[aux_index[0]] - curr_population[aux_index[1]] + curr_population[aux_index[2]] - curr_population[aux_index[3]])
    
    elif strat == "rand-to-best/1/bin" :

        aux_list = list(range(num_indiv))
        aux_list.pop(index_curr_indiv)

        cost_population = fun(curr_population, lims, spacing)
        index_best = np.argmin(cost_population)

        aux_index = random_state.choice(aux_list, 2, replace=False)
        mutant = curr_population[index_curr_indiv] + curr_population[index_best] \
                                                    + mut_factor * (curr_population[index_best] - curr_population[index_curr_indiv] \
                                                    + curr_population[aux_index[0]] - curr_population[aux_index[1]])
    
    elif strat == "rand-to-best/2/bin" :

        aux_list = list(range(num_indiv))
        aux_list.pop(index_curr_indiv)

        cost_population = fun(curr_population, lims, spacing)
        index_best = np.argmin(cost_population)

        aux_index = random_state.choice(aux_list, 4, replace=False)
        mutant = curr_population[index_curr_indiv] + curr_population[index_best] \
                                                    + mut_factor * (curr_population[index_best] - curr_population[index_curr_indiv] \
                                                    + curr_population[aux_index[0]] - curr_population[aux_index[1]] \
                                                    + curr_population[aux_index[2]] - curr_population[aux_index[3]])
    
    elif strat == "current-to-rand/1" :
        
        aux_list = list(range(num_indiv))
        aux_list.pop(index_curr_indiv)

        aux_index = random_state.choice(aux_list, 3, replace=False)
        mutant = curr_population[index_curr_indiv] + K_factor * (curr_population[aux_index[0]] - curr_population[index_curr_indiv]) \
                                                    + mut_factor * (curr_population[aux_index[1]] - curr_population[aux_index[2]])
    
    mutant = inside_bounds(mutant, lims, random_state)

    return mutant

def recomb(curr_indiv, mutant, strat, CR_prob, num_gene, random_state):

    """
    _________________
    --- Descrição ---
    _________________

    Função que aplica operação de recombinação entre genes de um indivíduo alvo e um mutante.

    __________________
    --- Argumentos ---
    __________________

    - curr_indiv : lista representando indivíduo em análise;

    - mutant : lista representando indivíduo mutante;

    - strat : string informando estratégia de mutação utilizada;

    - CR_prob : taxa de cross-over entre indivíduos;

    - num_gene : número de entradas do indivíduo;

    - random_state : uma seed do tipo np.RandomState para gereção de números aleatórios.

    ______________
    --- Saídas ---
    ______________

    - trial : lista representando indivíduo teste.

    """

    if strat[-3:] == "bin" :

        trial = np.zeros(num_gene)
        rand_vector = random_state.uniform(size=num_gene)
        rand_int = random_state.randint(low=0, high=num_gene)

        #Loop ao longo dos genes para compor o indivíduo teste.
        for k in range(num_gene):
            if rand_vector[k] <= CR_prob or k == rand_int: 
                trial[k] = mutant[k]
            else:
                trial[k] = curr_indiv[k]

        return trial

    elif strat[-1] == "1" :

        trial = mutant ** 1
        return trial

def select(curr_indiv, trial, fun, lims, spacing):

    """
    _________________
    --- Descrição ---
    _________________

    Função para ajustar um ponto no caso de alguma de suas entrdas não estiver contido dentro dos limites. Caso o ponto
    já respeite os limites, nada muda.

    __________________
    --- Argumentos ---
    __________________

    - curr_indiv : lista representando indivíduo em análise;

    - curr_indiv : lista representando indivíduo teste;

    - fun : função objetivo;

    - lims : lista representando os limites de busca;

    - spacing : espaçamento da malha computacional.

    ______________
    --- Saídas ---
    ______________

    - curr_indiv : lista representando o melhor indivíduo entre o atual e o teste.

    """

    
    local_cost = fun(curr_indiv, lims, spacing)
    trial_cost = fun(trial, lims, spacing)

    if local_cost > trial_cost:
        curr_indiv = trial ** 1

    return curr_indiv

def DE_evolve(index_curr_indiv, curr_population, fun, strat, mut_factor, K_factor, CR_prob, lims, spacing, num_indiv, num_gene, random_state):

    """
    _________________
    --- Descrição ---
    _________________

    Função que aplica a operação de evolução em um indivíduo.

    __________________
    --- Argumentos ---
    __________________

    - index_curr_indiv : índice do indivíduo alvo;

    - curr_population : matriz representando população da geração atual;

    - fun : função objetivo;

    - strat : string informando estratégia de mutação utilizada;

    -  mut_factor : coeficiente de mutação F;

    - K_factor : coeficiente de mutação K;

    - CR_prob : taxa de cross-over entre indivíduos;

    - lims : lista representando os limites de busca;

    - spacing : espaçamento da malha computacional;

    - num_indiv : número de indivíduos na população;

    - num_gene : número de entradas do indivíduo;

    - random_state : uma seed do tipo np.RandomState para gereção de números aleatórios.
    ______________
    --- Saídas ---
    ______________

    - next_indiv : lista representando o indivíduo selecionado para passar para a próxima geração. 

    """

    curr_indiv = curr_population[index_curr_indiv]

    mutant = mutate(index_curr_indiv, curr_population, fun, strat, mut_factor, K_factor, lims, spacing, num_indiv, random_state)

    trial = recomb(curr_indiv, mutant, strat, CR_prob, num_gene, random_state)

    next_indiv = select(curr_indiv, trial, fun, lims, spacing)

    return next_indiv

def DE_optimize(fun, strat, mut_factor, CR_prob, K_factor, lims, spacing, tol_gen, tol_delta, ini_population, num_indiv, num_gene, random_state=np.random.RandomState(0), hist_flag=False):

    """
    _________________
    --- Descrição ---
    _________________

    Função que aplica o algoritmo de Evolução Diferencial sobre uma função objetivo.

    __________________
    --- Argumentos ---
    __________________

    - fun : função objetivo;

    - strat : string informando estratégia de mutação utilizada;

    -  mut_factor : coeficiente de mutação F;

    - K_factor : coeficiente de mutação K;

    - CR_prob : taxa de cross-over entre indivíduos;

    - lims : lista representando os limites de busca;

    - tol_gen : tolerância de gereções a serem otimizadas;

    - tol_delta : tolerância da variação média entre os custos de duas gerações consecutivas;

    - ini_population : matriz representando população inicial;

    - spacing : espaçamento da malha computacional;

    - num_indiv : número de indivíduos na população;

    - num_gene : número de entradas do indivíduo;

    - random_state : uma seed do tipo np.RandomState para gereção de números aleatórios;

    - hist : valor booleano que aciona o armazenamento de histórico da população. Por padrão, falsa.
    ______________
    --- Saídas ---
    ______________

    - opt_indiv : lista representando indivíduo ótimo;
    
    - opt_cost : valor ótimo da função objetivo;
    
    - curr_population : matriz representando população final;
    
    - opt_cost_population : custo da população final;

    - hist : histórico de populções. Retornado apenas quando entrada hist = True.

    """

    curr_population = ini_population ** 1
    curr_cost = fun(curr_population, lims, spacing)

    if hist_flag:
        # hist = []
        hist = np.zeros((tol_gen, num_indiv, num_gene))

    cont_gen = 0; cost_delta = np.inf
    while cont_gen < tol_gen and cost_delta > tol_delta: 
        
        print(f"\r[DE/{strat}]: Generation {cont_gen+1}/{tol_gen}", end="")
            
        if hist_flag:
            # hist.append(curr_population)
            hist[cont_gen] = curr_population.copy()

        next_population = curr_population ** 1
           
        for j in range(num_indiv):
            
            next_population[j] = DE_evolve(j, curr_population, fun, strat, 
                                            mut_factor, K_factor, CR_prob, lims, spacing, 
                                            num_indiv, num_gene, random_state)

        next_cost = fun(next_population, lims, spacing)
        delta_cost = np.abs(np.sum(curr_cost-next_cost)/num_indiv)

        curr_population = next_population ** 1
        curr_cost = next_cost

        cont_gen += 1

    opt_cost_population = fun(curr_population, lims, spacing)
    aux_index = np.argmin(opt_cost_population)
    opt_indiv = curr_population[aux_index]
    opt_cost = opt_cost_population[aux_index]

    if hist_flag:
        return opt_indiv, opt_cost, curr_population, opt_cost_population, hist

    else:
        return opt_indiv, opt_cost, curr_population, opt_cost_population

def FaDE_optimize(fun, strat, mut_factor, CR_prob, K_factor, lims, spacing, tol_gen, tol_delta, ini_population, num_indiv, num_gene, random_state=np.random.RandomState(0), hist_flag=False, deffu=centroid):

    """
    _________________
    --- Descrição ---
    _________________

    Função que aplica o algoritmo de Evolução Diferencial adaptativo mediante sistemas lógicos fuzzy.

    __________________
    --- Argumentos ---
    __________________

    - fun : função objetivo;

    - strat : string informando estratégia de mutação utilizada;

    -  mut_factor : coeficiente inicial de mutação F;

    - K_factor : coeficiente de mutação K;

    - CR_prob : taxa inicial de cross-over entre indivíduos;

    - lims : lista representando os limites de busca;

    - tol_gen : tolerância de gereções a serem otimizadas;

    - tol_delta : tolerância da variação média entre os custos de duas gerações consecutivas;

    - ini_population : matriz representando população inicial;

    - spacing : espaçamento da malha computacional;

    - num_indiv : número de indivíduos na população;

    - num_gene : número de entradas do indivíduo;

    - random_state : uma seed do tipo np.RandomState para gereção de números aleatórios;

    - hist : valor booleano que aciona o armazenamento de histórico da população. Por padrão, falsa.
    ______________
    --- Saídas ---
    ______________

    - opt_indiv : lista representando indivíduo ótimo;
    
    - opt_cost : valor ótimo da função objetivo;
    
    - curr_population : matriz representando população final;
    
    - opt_cost_population : custo da população final;

    - hist : histórico de populções. Retornado apenas quando entrada hist = True.

    """

    curr_population = ini_population ** 1
    curr_cost = fun(curr_population, lims, spacing)

    F_val = mut_factor
    CR_val = CR_prob

    if hist_flag:
        # hist = []
        hist_CR, hist_F = [], []
        hist = np.zeros((tol_gen, num_indiv, num_gene))

    cont_gen = 0; cost_delta = np.inf
    while cont_gen < tol_gen and cost_delta > tol_delta: 
        
        print(f"\r[FaDE/{strat}]: Generation {cont_gen+1}/{tol_gen}", end="")
            
        if hist_flag:
            # hist.append(curr_population)
            hist_CR.append(CR_val)
            hist_F.append(F_val)
            hist[cont_gen] = curr_population.copy()

        next_population = curr_population ** 1
        for j in range(num_indiv):
            
            next_population[j] = DE_evolve(j, curr_population, fun, strat, 
                                            F_val, K_factor, CR_val, lims, spacing, 
                                            num_indiv, num_gene, random_state)

        next_cost = fun(next_population, lims, spacing)
        delta_cost = np.abs(np.sum(curr_cost-next_cost)/num_indiv)

        if cont_gen>1 :
            FC = FC_eval(next_cost, curr_cost, num_indiv)
            PC = PC_eval(next_population, curr_population, num_indiv, num_gene)
            d11, d12 = F_params_eval(FC, PC)
            d21, d22 = CR_params_eval(FC, PC)
            F_val = F_FLS(d11, d12, deffu)
            CR_val = CR_FLS(d21, d22, deffu)

        curr_cost = next_cost
        curr_population = next_population ** 1

        cont_gen += 1

    opt_cost_population = fun(curr_population, lims, spacing)
    aux_index = np.argmin(opt_cost_population)
    opt_indiv = curr_population[aux_index]
    opt_cost = opt_cost_population[aux_index]

    if hist_flag:
        return opt_indiv, opt_cost, curr_population, opt_cost_population, hist, hist_F, hist_CR

    else:
        return opt_indiv, opt_cost, curr_population, opt_cost_population
