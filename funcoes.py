# Fonte: https://www.pythonprogressivo.net/2018/07/import-Como-Criar-Importar-Usar-Modulo-Python-Curso.html

# Teste: imprime lista não ordenada de word_tag + valor
def print_normal(text):
    dic = count_word_tag(text)
    for key in list(dic.keys()):
        print(key, ":", dic[key])

# Teste: imprime lista ordenada
def print_sorted(text):
    dic = count_word_tag(text)
    for key in sorted(dic, key=dic.get):
        print(key, ":", dic[key])

# Teste: imprimir dicionário não ordenada
def print_dic(dic):
    for key in list(dic.keys()):
        print(key, ":", dic[key])

# Teste: imprimir valores do dicionário pela chave
def print_values(key, dic):
    if key in dic:
        values = dic[key]
        print(values)

# Teste: contar chaves
def count_key(key, dic):
    count = 0
    for chave in dic.keys():
        if chave == key:
            count = count + 1
            print(count)
            print('Chave:', chave, ', key:')
    return print(f'\nChave {chave} in dic possui count = {count}')

# salva dicionário em um arquivo
def save_dic_arq(dic, name_arq):
    arq = open(name_arq, "wt")
    dados = str(dic)    # converte dicionário em string
    arq.write(dados)
    arq.close()

# contagem das ocorrências de palavra_tags
"""
Faz a contagem das ocorrências de palavra_tag do arquivo de treino e retorna um dicionário de ocorrências.
Saída: Dicionário de Ocorrências:
Formato de saída: = {'876_cd': 1, '3.34_cd': 1, '2.59_cd': 1, 'kuse_nnp': 1, '46.125_cd': 1, 'temblor-prone_jj': 1}
"""
def count_word_tag(text):
    d = dict()

    # busca palavras no texto
    for line in text:
        line = line.strip()     # retorna cópia da string
        line = line.lower()     # converte a string em minúscula
        # adiciona nova lista de palavras na lista usando espaço como delimitador
        words = line.split(" ")

        # conta palavras e incrementa, se achou palavras repetidas
        for word in words:
            if word in d:
                d[word] = d[word] + 1
            else:
                d[word] = 1
    return d

# Verifica se a string é um numeral
def is_numeral(str):
    lista = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if str in lista:
        return True
    else:
        return False

# Verifica se a palavra é desconhecida
def is_unknown_word(key, dic):
    count = 0
    valores = dic.get(key)   # guarda valores de key
    
    for value in valores.values():    # percorre cada valor da palavra key
        count = count + value

    if count <= 5: return True  # é palavra desconhecida
    else: return False          # palavra conhecida

# separar chave, tag, valor e salvar em um dicionário (valor = dicionário)
# cria dicionário: extraindo a sequência de pares (tag, word) de cada sentença
# formato: palavras/chaves e suas tags/valores:


"""
Separa a palavra, tag e valor de um dicionário, que foi criado através do texto do corpus.
No dicionário criado, é salvo as correspondentes tags e valores de cada palavra.
Saída: Dicionário de Palavras, Tags e Ocorrências:
Formato de saída: Ex.: dicio_split = {'a': {'DT': 100, 'AR': 20}, 'the': {'AR': 2, 'PP': 25}}
"""
def split_key_tag_value(dic):

    dicio_split = {}    # dicionário: guarda chave, tag e valor
    dicio_tag = {}      # dicionário: guarda valores (dicionário: tag + valor)

    # busca e separa chave, tag e valor
    for chave, valor in dic.items():

        words = chave.split("_")     # lista: adiciona chave e tag usando '_' como delimitador

        # adiciona chave, tag e valor (valor = dicionário) no dicionário dicio_split
        if is_numeral(words[0][0]): # se a palavra for numeral
            words[0] = '#NUM'       # transforma numeral no token #NUM

        if words[0] in dicio_split:     # se a chave já existe
            dicio_tag = dicio_split.get(words[0])    # dicionário: acessar valores da chave
            dicio_tag[words[1]] = valor              # guarda valor da chave/tag
            dicio_split[words[0]] = dicio_tag        # guarda valores (tag + valor) da chave
        else:   # se a chave não existe
            dicio_tag[words[1]] = valor  # guarda chave e valor (tag + valor)
            dicio_split[words[0]] = dicio_tag   # guarda chave + valores (tag + valor)
        words.clear()    # apaga todos valores da lista
        dicio_tag = {}   # apaga todos valores do dicionário

    # dicionário: chaves e valores (tags + valores = dicionário)
    # Ex.: dicio_split = {'a': {'DT': 100, 'AR': 20}, 'the': {'AR': 2, 'PP': 25}}
    return dicio_split

# substitui palavras desconhecidas (aparecem até 5 vezes) por UNK
def replace_unknown_words(dic):
    for word, value in list(dic.items()):    # conversão dos itens do dicionário para a lista
        if is_unknown_word(word, dic):       # verifica se a palavra é desconhecida            
            dic['UNK'] = dic.pop(word)       # substitui a word por UNK

"""
Gera Dicionário de Testes: extrair a sequências de palavras de um corpus
Entrada: corpus de desenvolvimento (arquivo: Secs19-21 - development)
Formato de saída: Ex.: dicio_teste = ['the', 'Arizona', 'Corporations', 'Commission', (...)]
"""
def dicio_teste(arq_develop):
    lista = []  # guarda a sequência de palavras

    # busca palavras/tag no corpus
    for line in arq_develop:
        line = line.strip()     # retorna cópia da string
        line = line.lower()     # converte a string em minúscula
        # adiciona nova lista de palavras_tag na lista usando espaço como delimitador
        words_tag = line.split(" ")

        # separa word de tag
        for elemento in words_tag:
            # adiciona nova lista de palavras_tag na lista usando "_" como delimitador
            word = elemento.split("_")
            # adiciona nova palavra na lista
            lista.append(word[0])

    return lista

"""
Gera o Dicionário de Treino com as palavras, tags e as estimativas de probabilidades por contagem de ocorrências.
Calcula as probabilidades de tags em cada palavra (retorna palavra e tag com maior probabilidade)
Saída: Dicionário de Treino::
Formato de saída: Ex.: dicio_treino = {'deaths': {'nns': 29, 'prob': 1.0}, 'among': {'in': 397, 'prob': 1.0}, (...)}
"""
def dicio_treino(dic):
    
    dicio_max_word_tag_prob = {}    #dicionário: guarda chave, tag, valor/máximo e probabilidade

    for chave, valor in dic.items():

        max_key = max(valor, key=lambda key: valor[key])  # tag/chave máxima de cada palavra/chave
        dicio_tag_valor = {}
        dicio_tag_valor[max_key] = valor.get(max_key)     # guarda tag e valor da tag/chave máxima
        
        count_val = 0
        for tag, val in valor.items():
            count_val = count_val + val         # incrementa a soma de cada valor
        prob = valor.get(max_key)/count_val     # probabilidade de cada chave/tag (valor_tag / soma_valor_tags)
        dicio_tag_valor['prob'] = prob          # inserindo a probabilidade

        dicio_max_word_tag_prob[chave] = dicio_tag_valor    # inserindo: word e valor (dicionário: tag/max e valor/max + prob e valor)

    # dicionário: chaves e valores (dicionário: tags/max e valores/max + prob e valor)
    # Ex.: dicio_max_word_tag_prob = {'a': {'DT': 100, 'AR': 20}, 'the': {'AR': 2, 'PP': 25}}        
    return dicio_max_word_tag_prob


"""
Driver de Treino:
Entradas: lê Dicionário de Testes e Dicionário de Treino
Saída: gera arquivo de Lista Real com os pares word e tag
Formato de saída: Ex.: lista_real = ['the_dt', 'arizona_nnp', 'corporations_nns', 'commission_nnp', (...)]
"""
def driver_pos_tagger(dic_treino, dicio_teste):
    
    lista_real = []     # guarda os pares palavra_tag
    for word in dicio_teste:     # percorre o dicionário de teste
        #for chave, values in dic_treino.items():    # percorre o dicionário de treino
        if word in dic_treino:
            values = dic_treino.get(word)     # valores da chave 'word'
            #keys_val.keys()     # retona lista de chaves          
            keys_val = list(values.keys())            
            lista_real.append(word + '_' + keys_val[0])

        # substitui palavras desconhecidas (aparecem até 5 vezes) por UNK
        else:
            lista_real.append('UNK' + '_' + 'UNK')

    # Teste: imprime a lista de lista_real do teste
    #print('\n', lista_real)

    # salva o dicionário lista_real em arquivo
    save_dic_arq(lista_real, 'lista_real.txt')

    return lista_real

"""
            if chave == word:
                keys_val = list(values.keys())
                lista_real.append(word + '_' + keys_val[0])
                break
"""     
    



"""
gera arquivo de Lista de Tag Real contendo as tags de cada palavra
extrai, na sequência, as tags de cada palavra
Entrada: Lista Real
Saída: Lista de Tag Real contendo as tags de cada palavra na sequência
Formato de saída: Ex.: lista_tag_real = ['dt', 'nnp', 'nns, 'nnp]
"""
def listar_tag_real(lista_real):
    lista_tag_real = []
    for chave in lista_real:
        # adiciona nova lista de palavras na lista usando espaço como delimitador
        words = chave.split("_")
        lista_tag_real.append(words[1])
    # salva Lista de Tag Preditiva
    save_dic_arq(lista_tag_real, 'lista_tag_real_treino.txt')

    return lista_tag_real


"""
gera arquivo de Lista de Tag Preditiva contendo as tags de cada palavra
extrai, na sequência, as tags de cada palavra
Entrada: Arquivo corpus
Saída: Lista de Tag Preditiva contendo as tags de cada palavra na sequência
Formato de saída: Ex.: lista_tag_pred = [['dt', 'nnp', 'nns', (...)]
"""
def listar_tag_pred(arq_develop):
    lista = []  # guarda a sequência de palavras

    # busca palavras/tag no corpus
    for line in arq_develop:
        line = line.strip()     # retorna cópia da string
        line = line.lower()     # converte a string em minúscula
        # adiciona nova lista de palavras_tag na lista usando espaço como delimitador
        words_tag = line.split(" ")

        # separa word de tag
        for elemento in words_tag:
            # adiciona nova lista de palavras_tag na lista usando "_" como delimitador
            word = elemento.split("_")
            # adiciona nova tag na lista
            lista.append(word[1])
    # salva Lista de Tag Preditiva
    save_dic_arq(lista, 'lista_tag_pred_treino.txt')

    return lista

"""
Entrada: Lista de Tags Preditivo
Saída: lista de Rótulos
"""
def listar_labels(lista_tag_pred_treino):
    labels = []
    for tag in lista_tag_pred_treino:
        if len(labels) == 0:
            labels.append(tag)
        elif tag not in labels:
            labels.append(tag)
        else:
            test = 0
    return labels

