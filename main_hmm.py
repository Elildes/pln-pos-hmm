import funcoes
import glob
import re
import nltk
from nltk.corpus.reader import TaggedCorpusReader
from collections import defaultdict
from collections import Counter

# Observação: antes de rodar esta aplicação, veja as dependências no final deste arquivo.

# 1ª PARTE: 'Treino' (usando o corpus de treino; no caso do Penn Treebank usar Seções 0-18):

# Definição do HMM - Hidden Markov Model pelas Matrizes A (probabilidade de transição) e B (probabilidades de observação de palavras dadas tags)

'''
Matriz Aij de probabilidades P(ti|ti-1) de transição entre os estado ocultos (part-of-speech tags):
Possue linhas (i) são as tags anteriores e colunas (j) são as tags atuais.
A primeira linha da matriz A, <s>, corresponde ao estado inicial de cada tag atual (coluna j).
'''



'''
Matriz B de probabilidades de observação de palavras dadas tags:
Possue bi(ot), onde linhas (i) são as tags possíveis de cada word e colunas (ot) são as palavras/sequência de T_observações (texto de saída), extraídas de um vocabulário ??.
'''



'''
Matriz O é a rede de Virtebi que calcula a melhor sequência de estado oculto para a sequencia de observação do texto de saída.
Possue linhas (qi) que são as possíveis tags e colunas (oj) de n estados (words/saídas).
A matriz tem início na coluna 1 (pela 1ª word do texto) pelo valor (máximo) viterbi[s,t], calculado pelo algoritmo de Virtebi em cada posição da matriz,
através do produto do pi (conjunto das probabilidades do estado inicial qi, que foi pego da entrada <s>, primeira
linha da matriz A) e a probabilidade de observação da 1ª word do texto dado a tag para essa célula.
'''

'''
Tokenizar o corpus/texto
'''
#sentence = nltk.corpus.gutenberg.words( 'austen-emma.txt' )
#sentence = "At eight o'clock on Thursday morning Arthur didn't feel very good."
#tokens = nltk.word_tokenize(sentence)
#print(tokens)
#print(sentence)

#tagged = nltk.pos_tag(tokens)
#print(tagged[0:6])



test_sent = ["We",
            "have",
            "learned",
            "much",
            "about",
            "interstellar",
            "drives",
            "since",
            "a",
            "hundred",
            "years",
            "ago",
            "that",
            "is",
            "all",
            "I",
            "can",
            "tell",
            "you",
            "about",
            "them",
            ]

class pos_tagger():

    def __init__(self):
        self.unknown_prob = 0.0000000000001
        self.tagged_file = glob.glob("brown/*")
        self.bigram_cnt = {}
        self.unigram_cnt = {}
        self.tag_count = defaultdict(lambda: 0)
        self.tag_word_count = Counter()
        self.transition_probabilities = defaultdict(lambda: self.unknown_prob)
        self.emmission_probabilities = defaultdict(lambda: self.unknown_prob)

    def ngrams(self, text, n):
        Ngrams = []
        for i in range(len(text)): Ngrams.append(tuple(text[i: i + n]))
        return Ngrams

    def bigram_counts(self, tags):
        for i_tag_bigram in self.ngrams(tags, 2):
            if i_tag_bigram in self.bigram_cnt:
                self.bigram_cnt[i_tag_bigram] += 1
            else:
                self.bigram_cnt[i_tag_bigram] = 1
        return self.bigram_cnt

    def unigram_counts(self, tags):
        for tag in tags:
            if tag in self.unigram_cnt:
                self.unigram_cnt[tag] += 1
            else:
                self.unigram_cnt[tag] = 1
        return self.unigram_cnt

    def tag_word_counts(self, tagged_words):
        for tag, word in tagged_words:
            self.tag_count[tag] += 1
            if (word, tag) in self.tag_word_count:
                self.tag_word_count[(tag, word)] += 1
            else:
                self.tag_word_count[(tag, word)] = 1
        return self.tag_word_count

    def transition_probabilty(self, tags):
        bigrams = self.ngrams(tags, 2)
        for bigram in bigrams:
            self.transition_probabilities[bigram] = self.bigram_cnt[bigram] / self.unigram_cnt[bigram[0]]
        return self.transition_probabilities

    def emmission_probabilty(self, tagged_words):
        for tag, word in tagged_words:
            self.emmission_probabilities[tag, word] = self.tag_word_count[tag, word] / self.tag_count[tag]
        return self.emmission_probabilities

    def initial_probabilities(self, tag):
        return self.transition_probabilities["START", tag]

    def vertibi(self, observable, in_states):
        states = set(in_states)
        states.remove("START")
        states.remove("END")
        trails = {}
        for s in states:
            trails[s, 0] = self.initial_probabilities(s) * self.emmission_probabilities[s, observable[0]]
        # Run Viterbi when t > 0
        for o in range(1, len(observable)):
            obs = observable[o]
            for s in states:
                v1 = [(trails[k, o - 1] * self.transition_probabilities[k, s] * self.emmission_probabilities[s, obs], k) for k in states]
                k = sorted(v1)[-1][1]
                trails[s, o] = trails[k, o - 1] * self.transition_probabilities[k, s] * self.emmission_probabilities[s, obs]
        best_path = []
        for o in range(len(observable) - 1, -1, -1):
            k = sorted([(trails[k, o], k) for k in states])[-1][1]
            best_path.append((observable[o], k))
        best_path.reverse()
        for x in best_path:
            print(str(x[0]) + "," + str(x[1]))
        return best_path

    # muda as words em minpusculas
    def clean(self, word):
        word = re.sub('\s+', '', word.lower())      
        return word

    # Extrai as tags do corpus                    
    def tag(self):
        reader_corpus = TaggedCorpusReader('.', self.tagged_file)

        tagged_words = []   # lista de tags
        all_tags = []
        for sent in reader_corpus.tagged_sents():  # get tagged sentences
            all_tags.append("START")
            for (word, tag) in sent:
                if tag is None or tag in ['NIL']:
                    continue
                all_tags.append(tag)
                #word = self.clean(word)
                word = ""
                tagged_words.append((tag, word))
            all_tags.append("END")

        self.tag_word_counts(tagged_words)

        self.bigram_cnt = self.bigram_counts(all_tags)
        self.unigram_cnt = self.unigram_counts(all_tags)

        self.transition_probabilty(all_tags)
        self.emmission_probabilty(tagged_words)

        # Salva tagged_words e all_tags em arquivos
        funcoes.save_dic_arq(tagged_words, 'tagged_words.txt')
        funcoes.save_dic_arq(all_tags, 'all_tags.txt')

        # ERRO ????????????????????
        #self.tag_test(all_tags)

        # Testing
        with open("dicio_teste.txt", 'r') as arquivo_teste:
            dicio_teste = arquivo_teste.read()
            # TESTE
            #print(dicio_teste)
            #cleaned_test_sent = [self.clean(w) for w in dicio_teste]
            seq_pos_tagging_hmm = self.vertibi(list(dicio_teste), all_tags)
            funcoes.save_dic_arq(seq_pos_tagging_hmm, "seq_pos_tagging_hmm.txt")
            print(seq_pos_tagging_hmm)

ps = pos_tagger()
ps.tag()


"""
Lista de dependências:
Linux: 
. Python: sudo apt-get install python3: 
. Pip3: sudo apt-get install python3-pip
. Scikit-learn: pip install -U scikit-learn
. NLTK: https://www.nltk.org/install.html
Windows:
. Python: https://www.python.org/downloads/
. Pip 22.3: já incluso no python 3.11
. Scikit-learn: https://scikit-learn.org/dev/install.html
. NLTK: https://www.nltk.org/install.html
Obs.: instale nova dependência, caso seja solicitado 
"""