import collections
import itertools

MAX_SENTENCE_LENGTH = 3

class TaskDescriptionVectorizer(object):
    """
    Attributes:
        dictionary: token to index
        rev_dictionary: index to token
    
    Example:
        >>> vec = TaskDescriptionVectorizer(["'move right', 'move up', 'move left', 'move down',"])
        >>> task_desc_vec.transform(["move right", "move left"])
        [[1, 2, 0], [1, 4, 0]]

    """

    def __init__(
                self, 
                corpus, 
                max_sentence_length=MAX_SENTENCE_LENGTH,
            ):
        # word to index, if word does not exist, return 0 (UNK)
        self.dictionary = collections.defaultdict(int)
        # index to word, if index does not exist, return UNK
        self.rev_dictionary = collections.defaultdict(lambda: "UNK")
        self.max_sentence_length = max_sentence_length
        self.sentence_code_dim = None

        # Helpers
        def flat_map(func, x):
            return list(itertools.chain(*map(func, x)))

        # Preprocessing Functions
        def tokenize_line(line):
            return line.strip().split()

        def preprocessing_line(line):
            return line.lower()
        
        # workflow for preprocessing input
        self.preprocessing_actions = [
            preprocessing_line,
            tokenize_line,
        ]
        
        self.fit(flat_map(self.processing_one, corpus))

    def processing_one(self, line):
        tokens = line
        for func in self.preprocessing_actions:
            tokens = func(tokens)
        return tokens
    
    def transform_one(self, line):
        """
        Transform line to vector

        Note:
            - assume token number less than max sentence length
        """
        
        tokens = self.processing_one(line)
        while len(tokens) < self.max_sentence_length:
            tokens.append("UNK")
        return list(map(lambda token: self.dictionary[token], tokens))

    def transform(self, X):
        """
        Args:
            X: list of task descriptions
        
        Returns:
            vectors of task descriptions
        """
        return list(map(self.transform_one, X))
    
    def fit(self, tokens):
        """
        Build dictionary (token to index), rev_dictionary (index to token)
        """
        count = collections.Counter(tokens)
        self.sentence_code_dim = len(count)

        ordered_tokens = ['UNK'] + sorted(count.keys(), key=lambda k: count[k], reverse=True)

        self.dictionary = collections.defaultdict(int)
        self.rev_dictionary = collections.defaultdict(lambda: "UNK")

        for i, word in enumerate(ordered_tokens):
            self.dictionary[word] = i
            self.rev_dictionary[i] = word