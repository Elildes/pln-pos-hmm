# Modelagem de Sequence Labeling Problema para NLP como HMM - Hidden Markov Model  

## Dependências  

Antes de rodar esta aplicação, instale as dependências abaixo:  

**Lista de dependências:**  

Linux:  
1. Python: sudo apt-get install python3  
2. Pip3: sudo apt-get install python3-pip  
3. Scikit-learn: pip install -U scikit-learn  
4. NLTK: https://www.nltk.org/install.html  

Windows:  
1. Python: https://www.python.org/downloads/  
2. Pip 22.3: já incluso no python 3.11  
3. Scikit-learn: https://scikit-learn.org/dev/install.html  
4. NLTK: https://www.nltk.org/install.html  

**Obs.**: instale nova dependência, caso seja solicitado.  

## Definição do HMM - Hidden Markov Model pelas Matrizes A (probabilidade de transição) e B (probabilidades de observação de palavras dadas tags)  

Matriz Aij de probabilidades P(ti|ti-1) de transição entre os estado ocultos (part-of-speech tags):  
Possue linhas (i) são as tags anteriores e colunas (j) são as tags atuais.  
A primeira linha da matriz A, <s>, corresponde ao estado inicial de cada tag atual (coluna j).  

Matriz B de probabilidades de observação de palavras dadas tags:  
Possue bi(ot), onde linhas (i) são as tags possíveis de cada word e colunas (ot) são as palavras/sequência de T_observações (texto de saída), extraídas de um vocabulário ??.  

Matriz O é a rede de Virtebi que calcula a melhor sequência de estado oculto para a sequencia de observação do texto de saída.  
Possue linhas (qi) que são as possíveis tags e colunas (oj) de n estados (words/saídas).  
A matriz tem início na coluna 1 (pela 1ª word do texto) pelo valor (máximo) viterbi[s,t], calculado pelo algoritmo de Virtebi em cada posição da matriz, através do produto do pi (conjunto das probabilidades do estado inicial qi, que foi pego da entrada <s>, primeira linha da matriz A) e a probabilidade de observação da 1ª word do texto dado a tag para essa célula.  
