import ast
from sklearn.model_selection import train_test_split
from flair.data import Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, TransformerWordEmbeddings
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


columns = {0: 'text', 1: 'pos'}

data_folder = '/home/colin/dbbe_data/'  # directory containing the data files

corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='pos_flair_train.txt',
                              dev_file='pos_dev.txt',
                              test_file='pos_new_test.txt')

tag_type = 'pos'

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

embedding_types = [
    TransformerWordEmbeddings("/home/colin/pos_taggers/bert_12_agbgmg/lm_bert_12_agbgmg/"),  # path to transformer embeddings
    CharacterEmbeddings()
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('/home/colin/pos_taggers/pos_bert_12_agbgmg/POS_TAGGER_ANCIENT_GREEK_final',  # output path
              learning_rate=0.1,
              mini_batch_size=8,
              max_epochs=5,
              monitor_train=False,
              monitor_test=False,
              train_with_dev=True)
