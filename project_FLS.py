import numpy as np

def FC_eval(f_curr, f_prev, num_indiv):

    """
    _________________
    --- Descrição ---
    _________________

    Calcula a variação média quadrática entre os custos da função objetivo em duas gerações consecutivas,
    servindo como entrada fuzzy para o sistema de controle adaptativo de parâmetros.

    __________________
    --- Argumentos ---
    __________________

    - f_curr : vetor representando os custos da população atual;

    - f_prev : vetor representando os custos da população anterior;

    - num_indiv : número de indivíduos na população.

    ______________
    --- Saídas ---
    ______________

    - fc : valor escalar representando a variação média entre os custos (Function Change - FC).
    """
    
    aux0 = f_curr - f_prev
    aux1 = aux0**2
    aux2 = np.sum(aux1)
    aux3 = (1/num_indiv)*aux2
    
    return np.sqrt(aux3)

def PC_eval(pop_curr, pop_prev, num_indiv, num_gene):

    """
    _________________
    --- Descrição ---
    _________________

    Calcula a variação média quadrática entre posições da população atual e da anterior,
    caracterizando o deslocamento coletivo da população ao longo do espaço de busca.

    __________________
    --- Argumentos ---
    __________________

    - pop_curr : matriz contendo os indivíduos da geração atual (shape: [num_indiv, num_gene]);

    - pop_prev : matriz contendo os indivíduos da geração anterior (shape: [num_indiv, num_gene]);

    - num_indiv : número de indivíduos na população;

    - num_gene : número de genes (variáveis de decisão) por indivíduo.

    ______________
    --- Saídas ---
    ______________

    - pc : valor escalar representando a variação média entre populações (Population Change - PC).
    """

    aux0 = pop_curr - pop_prev
    aux1 = aux0*aux0
    aux2 = np.sum(aux1, axis=1)
    aux3 = (1/num_indiv)*np.sum(aux2)
    
    return np.sqrt(aux3)

def F_params_eval(FC, PC):

    """
    _________________
    --- Descrição ---
    _________________

    Gera os parâmetros fuzzy de entrada para o sistema de inferência do parâmetro F,
    aplicando funções não-lineares de normalização sobre as variáveis de variação.

    __________________
    --- Argumentos ---
    __________________

    - FC : escalar representando a variação de custo entre gerações;

    - PC : escalar representando a variação de população entre gerações.

    ______________
    --- Saídas ---
    ______________

    - d11 : entrada fuzzy normalizada a partir de PC;

    - d12 : entrada fuzzy normalizada a partir de FC.
    """

    aux0 = 1 - (1 + PC)*np.exp(-PC)
    aux1 = 1 - (1 + FC)*np.exp(-FC)

    return aux0, aux1

def CR_params_eval(FC, PC):

    """
    _________________
    --- Descrição ---
    _________________

    Gera os parâmetros fuzzy de entrada para o sistema de inferência do parâmetro CR,
    utilizando um mapeamento não-linear sobre as variações entre gerações.

    __________________
    --- Argumentos ---
    __________________

    - FC : escalar representando a variação de custo entre gerações;

    - PC : escalar representando a variação de população entre gerações.

    ______________
    --- Saídas ---
    ______________

    - d21 : entrada fuzzy normalizada a partir de PC;

    - d22 : entrada fuzzy normalizada a partir de FC.
    """

    aux0 = 2*(1 - (1 + PC)*np.exp(-PC))
    aux1 = 2*(1 - (1 + FC)*np.exp(-FC))

    return aux0, aux1

def G(sig, mu):

    """
    _________________
    --- Descrição ---
    _________________

    Gera uma função de pertinência gaussiana baseada nos parâmetros de dispersão e centro,
    a ser utilizada nos sistemas fuzzy como função de entrada ou de saída.

    __________________
    --- Argumentos ---
    __________________

    - sig : desvio padrão da função gaussiana;

    - mu : média (ou centro) da função gaussiana.

    ______________
    --- Saídas ---
    ______________

    - aux : função escalar que avalia a pertinência de um valor x segundo a distribuição gaussiana.
    """

    def aux(x):
        return np.exp(-((x-mu)**2)/(2*sig**2))
    
    return aux

### Definem-se os conjuntos fuzzy do Dicionário para o Sistema de Regras ###

d11_S = G(0.25, 0.05)
d11_M = G(0.25, 0.5)
d11_B = G(0.25, 0.9)

d12_S = G(0.35, 0.01)
d12_M = G(0.35, 0.5)
d12_B = G(0.35, 0.9)

F_S = G(0.5, 0.3)
F_M = G(0.5, 0.6)
F_B = G(0.5, 0.9)


d21_S = G(0.5, 0.1)
d21_M = G(0.5, 0.8)
d21_B = G(0.5, 1.5)

d22_S = G(0.5, 0.1)
d22_M = G(0.5, 0.8)
d22_B = G(0.5, 1.5)

CR_S = G(0.35, 0.4)
CR_M = G(0.35, 0.7)
CR_B = G(0.35, 1.0)


def F_rules(d11, d12):

    """
    _________________
    --- Descrição ---
    _________________

    Aplica o conjunto de regras fuzzy para o parâmetro F com base na combinação de pertinência
    das entradas d11 e d12, retornando a função de saída fuzzy correspondente.

    __________________
    --- Argumentos ---
    __________________

    - d11 : função de pertinência aplicada à entrada d11;

    - d12 : função de pertinência aplicada à entrada d12.

    ______________
    --- Saídas ---
    ______________

    - F : função de pertinência de saída fuzzy para o parâmetro F.
    """

    if d11==d11_S and d12==d12_S :
        F = F_S
    if d11==d11_S and d12==d12_M:
        F = F_M
    if d11==d11_S and d12==d12_B :
        F = F_B
    if d11==d11_M and d12==d12_S :
        F = F_M
    if d11==d11_M and d12==d12_M :
        F = F_M
    if d11==d11_M and d12==d12_B :
        F = F_B
    if d11==d11_B and d12==d12_S :
        F = F_B
    if d11==d11_B and d12==d12_M :
        F = F_B
    if d11==d11_B and d12==d12_B :
        F = F_B

    return F

def CR_rules(d21, d22):

    """
    _________________
    --- Descrição ---
    _________________

    Aplica o conjunto de regras fuzzy para o parâmetro CR com base na combinação de pertinência
    das entradas d21 e d22, retornando a função de saída fuzzy correspondente.

    __________________
    --- Argumentos ---
    __________________

    - d21 : função de pertinência aplicada à entrada d21;

    - d22 : função de pertinência aplicada à entrada d22.

    ______________
    --- Saídas ---
    ______________

    - CR : função de pertinência de saída fuzzy para o parâmetro CR.
    """

    if d21==d21_S and d22==d22_S :
        CR = CR_S
    if d21==d21_S and d22==d22_M:
        CR = CR_M
    if d21==d21_S and d22==d22_B :
        CR = CR_B
    if d21==d21_M and d22==d22_S :
        CR = CR_M
    if d21==d21_M and d22==d22_M :
        CR = CR_M
    if d21==d21_M and d22==d22_B :
        CR = CR_B
    if d21==d21_B and d22==d22_S :
        CR = CR_B
    if d21==d21_B and d22==d22_M :
        CR = CR_B
    if d21==d21_B and d22==d22_B :
        CR = CR_B
    
    return CR

def F_inference(d11, d12):

    """
    _________________
    --- Descrição ---
    _________________

    Executa a inferência fuzzy Mamdani para o parâmetro F, combinando todos os pares possíveis
    de regras com operações de mínimo e máxima agregação.

    __________________
    --- Argumentos ---
    __________________

    - d11 : valor escalar normalizado da primeira entrada fuzzy;

    - d12 : valor escalar normalizado da segunda entrada fuzzy.

    ______________
    --- Saídas ---
    ______________

    - space : domínio discreto do conjunto de saída F;

    - infer : função de pertinência agregada resultante da inferência Mamdani.
    """

    d11_part = [d11_S, d11_M, d11_B]
    d12_part = [d12_S, d12_M, d12_B]
    infer = []
    space = np.linspace(0, 2, 100)

    for rule1 in d11_part:
        for rule2 in d12_part:
            w = np.minimum(rule1(d11), rule2(d12))
            F = F_rules(rule1, rule2)
            infer.append(np.minimum(w, F(space)))

    return space, np.max(np.stack(infer), axis=0)

def CR_inference(d21, d22):

    """
    _________________
    --- Descrição ---
    _________________

    Executa a inferência fuzzy Mamdani para o parâmetro CR, combinando todos os pares possíveis
    de regras com operações de mínimo e máxima agregação.

    __________________
    --- Argumentos ---
    __________________

    - d21 : valor escalar normalizado da primeira entrada fuzzy;

    - d22 : valor escalar normalizado da segunda entrada fuzzy.

    ______________
    --- Saídas ---
    ______________

    - space : domínio discreto do conjunto de saída CR;

    - infer : função de pertinência agregada resultante da inferência Mamdani.
    """

    d21_part = [d21_S, d21_M, d21_B]
    d22_part = [d22_S, d22_M, d22_B]
    infer = []
    space = np.linspace(0, 1, 100)

    for rule1 in d21_part:
        for rule2 in d22_part:
            w = np.minimum(rule1(d21), rule2(d22))
            CR = CR_rules(rule1, rule2)
            infer.append(np.minimum(w, CR(space)))

    return space, np.max(np.stack(infer), axis=0)

def centroid(x, phi_x):

    """
    _________________
    --- Descrição ---
    _________________

    Realiza a defuzzificação do conjunto fuzzy utilizando o método do centróide
    (center of gravity), calculando o centro de massa da função de pertinência.

    __________________
    --- Argumentos ---
    __________________

    - x : vetor com os valores discretos do domínio;

    - phi_x : vetor com os graus de pertinência correspondentes em x.

    ______________
    --- Saídas ---
    ______________

    Escalar representando o valor crisp obtido por centróide.
    """

    return np.sum(x*phi_x)/np.sum(phi_x)

def mean_max(x, phi_x):

    """
    _________________
    --- Descrição ---
    _________________

    Realiza a defuzzificação pela média aritmética dos valores em que a pertinência
    atinge seu valor máximo.

    __________________
    --- Argumentos ---
    __________________

    - x : vetor com os valores discretos do domínio;

    - phi_x : vetor com os graus de pertinência correspondentes em x.

    ______________
    --- Saídas ---
    ______________

    Valor médio dos x para os quais phi_x é máximo.
    """

    max_val = np.max(phi_x)
    max_index = np.argwhere(phi_x == max_val)
    return np.sum(x[max_index])/len(max_index)

def center_max(x, phi_x):

    """
    _________________
    --- Descrição ---
    _________________

    Realiza a defuzzificação calculando o ponto médio do intervalo formado pelos
    valores de x onde a pertinência atinge o máximo.

    __________________
    --- Argumentos ---
    __________________

    - x : vetor com os valores discretos do domínio;

    - phi_x : vetor com os graus de pertinência correspondentes em x.

    ______________
    --- Saídas ---
    ______________

    Ponto médio do intervalo de máximos.
    """

    max_val = np.max(phi_x)
    max_index = np.argwhere(phi_x == max_val)
    inf_x = np.min(x[max_index])
    sup_x = np.max(x[max_index])
    return (inf_x + sup_x)*0.5

def F_FLS(d11, d12, deffu=centroid):
    
    """
    _________________
    --- Descrição ---
    _________________

    Aplica um sistema de inferência fuzzy Mamdani para determinar o valor do
    parâmetro F a partir das entradas normalizadas d11 e d12, com defuzzificação configurável.

    __________________
    --- Argumentos ---
    __________________

    - d11 : entrada fuzzy referente à variação populacional (PC);

    - d12 : entrada fuzzy referente à variação de custo (FC);

    - deffu : função de defuzzificação a ser utilizada (padrão: centróide).

    ______________
    --- Saídas ---
    ______________

    - F_new : valor defuzzificado para o parâmetro F.
    """

    space, infer = F_inference(d11, d12)
    F_new = deffu(space, infer)

    return F_new

def CR_FLS(d21, d22, deffu=centroid):

    """
    _________________
    --- Descrição ---
    _________________

    Aplica um sistema de inferência fuzzy Mamdani para determinar o valor do
    parâmetro CR a partir das entradas normalizadas d21 e d22, com defuzzificação configurável.

    __________________
    --- Argumentos ---
    __________________

    - d21 : entrada fuzzy referente à variação populacional (PC);

    - d22 : entrada fuzzy referente à variação de custo (FC);

    - deffu : função de defuzzificação a ser utilizada (padrão: centróide).

    ______________
    --- Saídas ---
    ______________

    - CR_new : valor defuzzificado para o parâmetro CR.
    """

    space, infer = CR_inference(d21, d22)
    CR_new = deffu(space, infer)

    return CR_new

