import math

class Handle_token:
    def __init__(self, tokens):
        self.dict_size = len(tokens)
        self.token_id = {}
        self.id_token = {}
        for idx, word in enumerate(tokens):
            self.token_id[word] = idx
            self.id_token[idx] = word
        self.start_id = self.token_id["[START]"]
        self.end_id = self.token_id["[END]"]
        self.none_id = self.token_id["[NONE]"]
        self.pad_id = self.token_id["[PAD]"]
    def id_to_token(self, token_id):
        return self.id_token.get(token_id)
    def token_to_id(self, token):
        return self.token_id.get(token, self.none_id)
    def encode(self, tokens):
        token_ids = [self.start_id, ]
        for token in tokens:
            token_ids.append(self.token_to_id(token))
        token_ids.append(self.end_id)
        return token_ids
    def decode(self, token_ids):
        flag_tokens = {"[START]", "[END]"}
        tokens = []
        for idx in token_ids:
            token = self.id_to_token(idx)
            if token not in flag_tokens:
                tokens.append(token)
        return tokens
class PoetryDataSet:
    def __init__(self, data, Handle_token, batch_size):
        self.data = data
        self.total_size = len(self.data)
        self.Handle_token = Handle_token
        self.batch_size = BATCH_SIZE
        self.steps = int(math.floor(len(self.data) / self.batch_size))
    def pad_line(self, line, length, padding=None):
        if padding is None:
            padding = self.Handle_token.pad_id
        padding_length = length - len(line)
        if padding_length > 0:
            return line + [padding] * padding_length
        else:
            return line[:length]
    def __len__(self):
        return self.steps
    def __iter__(self):
        np.random.shuffle(self.data)
        for start in range(0, self.total_size, self.batch_size):
            end = min(start + self.batch_size, self.total_size)
            data = self.data[start:end]
            max_length = max(map(len, data))
            batch_data = []
            for str_line in data:
                encode_line = self.Handle_token.encode(str_line)
                pad_encode_line = self.pad_line(encode_line, max_length + 2)
                batch_data.append(pad_encode_line)
            batch_data = np.array(batch_data)
            yield batch_data[:, :-1], batch_data[:, 1:]
    def generator(self):
        while True:
            yield from self.__iter__()
