import json
from collections import defaultdict, Counter
import re

class Captions():
    def __init__(self, captions_file):
        """
            Args:
            captions_file - coco captions json file
        """
        # captions_file = captions json file
        self.captions = defaultdict(list)
        self._cap_json = captions_file
        self.max_length = 0
        self._fn_to_id = None
        self._load_captions_from_file()
        # maintain separate dictionary with indexed captions, look at memory usage
        self.captions_indexed = self.captions.copy()

    def _load_captions_from_file(self):
        with open(self._cap_json) as rf:
            try:
                j = json.loads(rf.read())
            except FileNotFoundError as e:
                raise
            imid_fn = {img['id']:img['file_name'] for img in j['images']}
            self._fn_to_id = {img['file_name']:img['id'] for img in j['images']}
            # for every file name 5 captions
            for cap in j['annotations']:
                tokenized = self._tokenize_caption(cap['caption'])
                seq_length = len(tokenized)
                if seq_length > self.max_length:
                    self.max_length = seq_length
                self.captions[imid_fn[cap['image_id']]].append(tokenized)

    def _tokenize_caption(self, caption):
        return ['<BOS>'] + list(
            filter(lambda x: len(x) > 0, re.split(
                r'\W+', caption.lower()))) + ['<EOS>']

    def index_captions(self, word2idx):
        '''
        Args:
            word2idx: word to indices mapping
        '''
        def add_index(word):
            try:
                index = word2idx[word]
            except KeyError as e:
                index = word2idx['<UNK>']
            return index
        # take caption and tokenize it
        for name in self.captions_indexed:
            for i in range(len(self.captions_indexed[name])):
                # if word not in vocab, use <UNK>
                self.captions_indexed[name][i] = [add_index(word)
                                                  for word in
                                                  self.captions_indexed[name][i]]
    @property
    def filename_to_imid(self):
        return self._fn_to_id

# building Vocabulary
class Dictionary(object):
    def __init__(self, caption_dict):
        """
        Args:
            caption_dict: dictionary of {file_name: [['cap1'], ['cap2']]}
        """
        # sentences - array of sentences
        self._captions = caption_dict
        self._word2idx = {}
        self._idx2word = {}
        self._words = []
        self._get_words()
        # add tokens
        self._words.append('<UNK>')
        self.build_vocabulary()

    @property
    def vocab_size(self):
        return len(self._idx2word)

    @property
    def word2idx(self):
        return self._word2idx

    @property
    def idx2word(self):
        return self._idx2word

    def seq2dx(self, sentence):
        return [self.word2idx[wd] for wd in sentence]

    def _get_words(self):
        for image in self._captions:
            for cap in self._captions[image]:
                for word in cap:
                    word = word if word in ["<EOS>",
                                            "<BOS>",
                                            "<PAD>"] else word.lower()
                    self._words.append(word)

    def build_vocabulary(self):
        counter = Counter(self._words)
        # words, that occur less than 5 times dont include
        sorted_dict = sorted(counter.items(), key= lambda x: (-x[1], x[0]))
        sorted_dict = [(wd, count) for wd, count in sorted_dict
                       if count >= 3 or wd == '<UNK>']
        # after sorting the dictionary, get ordered words
        words, _ = list(zip(*sorted_dict))
        self._word2idx = dict(zip(words, range(1, len(words) + 1)))
        self._idx2word = dict(zip(range(1, len(words) + 1), words))
        # add <PAD> as zero
        self._idx2word[0] = '<PAD>'
        self._word2idx['<PAD>'] = 0
        print("Vocabulary size: ", len(self._word2idx))

    def __len__(self):
        return len(self.idx2word)
