class Vocab:
    UNK = "UUUNKKK"
    START = "<s>"
    END = "</s>"
    PAD_DUMMY = "PAD_DUMMY"  # TODO maybe to reomve
    PAD_IDX = 0  # TODO maybe to reomve
    START_IDX = 1
    END_IDX = 2
    def __init__(self, train_file):
        self.train_file = train_file
        self.tokens = self.get_unique()
        self.tokens.insert(self.PAD_IDX, self.PAD_DUMMY)
        self.tokens.insert(self.START_IDX, self.START)
        self.tokens.insert(self.END_IDX, self.END)

        self.vocab_size = len(self.tokens)
        self.i2token = {i: w for i, w in enumerate(self.tokens)}
        self.token2i = {w: i for i, w in self.i2token.items()}

    def get_word_index(self, word):
        if word in self.token2i:
            return self.token2i[word]
        return self.token2i[self.UNK]

    def get_unique(self):
        tokens = []
        token_seen = set()
        with open(self.train_file) as f:
            lines = f.readlines()
        for line in lines:
            if line == "" or line == "\n":
                continue
            for t in line.strip().split():
                if t not in token_seen:
                    token_seen.update(t)
                    tokens.append(t)
        tokens.append(self.UNK)
        return tokens


