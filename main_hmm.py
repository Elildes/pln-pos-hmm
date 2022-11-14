import funcoes
import glob
import re
import nltk
from nltk.corpus.reader import TaggedCorpusReader
from collections import defaultdict
from collections import Counter
from nltk.tokenize import word_tokenize

'''
Classe principal que imtegra todas os métodoa para a execução da aplicação.
'''
class pos_tagger():

    # Construtor que irá iniciar as variáveis.
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

    # Calcula a probabilidade de transição das tags
    def transition_probabilty(self, tags):
        bigrams = self.ngrams(tags, 2)
        for bigram in bigrams:
            self.transition_probabilities[bigram] = self.bigram_cnt[bigram] / self.unigram_cnt[bigram[0]]
        return self.transition_probabilities

    # Calcula a probabilidade de emissão da word/tag
    def emmission_probabilty(self, tagged_words):
        for tag, word in tagged_words:
            self.emmission_probabilities[tag, word] = self.tag_word_count[tag, word] / self.tag_count[tag]
        return self.emmission_probabilities

    # Calcula a probabilidade incial de um tag ser a primeira na sentença
    def initial_probabilities(self, tag):
        return self.transition_probabilities["START", tag]

    '''
    Algoritmo de Viterbi encontra a probabilidade mais alta dada para uma palavra em todas as nossas
    tags examinando nossas probabilidades de transmissão e emissão, multiplicando as probabilidades e,
    em seguida, encontrando a probabilidade máxima.
    '''
    def viterbi(self, observable, in_states):
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
        lista_pred_tag = []
        for x in best_path:
            #print(str(x[0]) + "," + str(x[1]))
            lista_pred_tag.append(str(x[1]))    # salvar as tags na lista
        funcoes.save_dic_arq(lista_pred_tag, "lista_pred_tag.txt")  # salvar a lista de tags preditiva em arquivo
        return best_path

    # muda as words em minúsculas
    def clean(self, word):
        word = re.sub('\s+', '', word.lower())      
        return word

    """
    Gera Dicionário de Testes: extrair a sequências de palavras de um corpus (words_tag)
    Entrada: lista = ['The/at', 'Fulton/np-tl',  'County/nn-tl', ...]
    Formato de saída: Ex.: dicio_teste = ['the', 'Fulton', 'County', (...)]
    """
    def split_dicio_teste(self, dicio, teste):
        lista_words = []  # guarda a sequência de words
        lista_tags = []    # guarda a sequência de tags

        # busca palavras/tag no dicio
        for pos in dicio:
            pos = pos.strip()     # retorna cópia da string
            pos = pos.lower()     # converte a string em minúscula
            # separa word de tag: adiciona nova lista de words e tag na lista usando '/' como delimitador
            words_tag = pos.split("/")

            if len(words_tag) == 2:
                lista_words.append(words_tag[0])   # salva as words na lista
                lista_tags.append(words_tag[1])    # salva as tags na lista
            else: continue

        if (teste == "word"): return lista_words
        if (teste == "tag"): return lista_tags
        else: return []

    # Treino do Brown Corpus
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
                word = self.clean(word)
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

        # Testing:
        # Abre o arquivos corpus e salva o conteúdo numa lista/array (dicionário de testes)
        # Saída: texto = [word1/tag1 word2/tag2 ...]
        dicio_teste = []
        with open("ca03.txt", "r") as file:    
            dicio_teste = file.read()

        # Testes
        dicio_teste = "The/at Fulton/np-tl County/nn-tl Grand/jj-tl Jury/nn-tl said/vbd Friday/nr an/at investigation/nn of/in Atlanta's/np$ recent/jj primary/nn election/nn produced/vbd ``/`` no/at evidence/nn ''/'' that/cs any/dti irregularities/nns took/vbd place/nn ./. The/at jury/nn further/rbr said/vbd in/in term-end/nn presentments/nns that/cs the/at City/nn-tl Executive/jj-tl Committee/nn-tl ,/, which/wdt had/hvd over-all/jj charge/nn of/in the/at election/nn ,/, ``/`` deserves/vbz the/at praise/nn and/cc thanks/nns of/in the/at City/nn-tl of/in-tl Atlanta/np-tl ''/'' for/in the/at manner/nn in/in which/wdt the/at election/nn was/bedz conducted/vbn ./."

        # Tokeniza o dicinário de testes.
        # Entrada: texto = [word1/tag1 word2/tag2 ...]
        # Saída: dicio = ['word1/tag1', 'word2/tag2', ...]
        words_tags_tokens = word_tokenize(dicio_teste)
        #print(words_tags_tokens)

        # Tokeniza em words ou tags o dicionário de testes.
        # Saída 01: dicio_word = ['word1', 'word2, ...]
        # Saída 02: lista_tags - ['tag1', 'tag2', 'tag3', ...]
        dicio_teste_words = self.split_dicio_teste(words_tags_tokens, "word")
        lista_real_tag = self.split_dicio_teste(words_tags_tokens, "tag")

        # Teste:
        #dicio_teste_words = ['the', 'fulton', 'county', 'grand', 'jury', 'said', 'friday', 'an', 'investigation', 'of']
        #dicio_teste_words = ['several', 'defendants', 'in', 'the', 'summerdale', 'police', 'burglary', 'trial', 'made', 'statements']

        
        # salva o dicionário de teste e lista real de tag em arquivo
        funcoes.save_dic_arq(dicio_teste_words, 'dicio_teste_words.txt')
        funcoes.save_dic_arq(lista_real_tag, 'lista_real_tag.txt')

        # Teste: Imprime lista de words e tags
        #print(dicio_teste_words)
        #print(lista_real_tag)

        # POS tagging usando HMM e Algoritmo de Viterbi
        # Entrada: ['word1', 'word2, ...]
        # Saída: lista = ['The/at', 'Fulton/np-tl', 'County/nn-tl', 'Grand/jj-tl', ...]
        seq_pos_tagging_hmm = self.viterbi(dicio_teste_words, all_tags)
        funcoes.save_dic_arq(seq_pos_tagging_hmm, "seq_pos_tagging_hmm.txt")
        #print(seq_pos_tagging_hmm)

        print("Fim do programa!!")


ps = pos_tagger()
ps.tag()
