import funcoes

# Observação: antes de rodar esta aplicação, veja as dependências no final deste arquivo.

# 1ª PARTE: 'Treino' (usando o corpus de treino; no caso do Penn Treebank usar Seções 0-18):

# abre arquivo de texto: corpus de treino
dir = r'Secs0-18-training.txt'
aqq_treino = open(dir, 'r')

# Teste: imprime lista de word_tag + valor (unigrama)
#funcoes.print_normal(aqq_treino)

# Teste: imprime lista ordenada de word_tag + valor
#funcoes.print_sorted(aqq_treino)


# Definição do HMM - Hidden Markov Model pelas Matrizes A (probabilidade de transição) e B (probabilidades de observação de palavras dadas tags)

'''
Matriz Aij de probabilidades P(ti|ti-1) de transição entre os estado ocultos (part-of-speech tags):
linhas (i) são as tags anteriores e colunas (j) são as tags atuais.
'''



'''
Matriz B de probabilidades de observação de palavras dadas tags:
bi(ot), onde linhas (i) são as tags possíveis de cada word e colunas (ot) são as palavras/sequência de T_observações (texto de saída), extraídas de um vocabulário ??.
'''



'''
Matriz O é a rede de Virtebi que calcula a melhor sequência de estado oculto para a sequencia de observação so texto de saída.
colunas n estados (words/saídas) 
'''








"""
Lista de dependências:
Linux: 
. Python: sudo apt-get install python3: 
. Pip3: sudo apt-get install python3-pip
. Scikit-learn: pip install -U scikit-learn
Windows:
. Python: https://www.python.org/downloads/
. Pip 22.3: já incluso no python 3.11
. Scikit-learn: https://scikit-learn.org/dev/install.html
Obs.: instale nova dependência, caso seja solicitado 
"""