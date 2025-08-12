import tensorflow as tf
import numpy as np
'''
 1. 새로운 token 등장 시 ID 부여
 2. embedding matrix 확장
 3. ID <-> token mapping 저장
'''
class DynamicTokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = []
        self.vocab_size = 0

    def encode(self, tokens):
        ids = []
        for token in tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = self.vocab_size
                self.id_to_token.append(token)
                self.vocab_size += 1
            ids.append(self.token_to_id[token])
        return ids

    def decode(self, ids):
        return [self.id_to_token[i] for i in ids if i < len(self.id_to_token)]

    def get_vocab_size(self):
        return self.vocab_size


class EmbeddingManager:
    def __init__(self, embedding_dim=16):
        self.tokenizer = DynamicTokenizer()
        self.embedding_dim = embedding_dim
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=1, output_dim=embedding_dim)

    def update_embeddings(self):
        vocab_size = self.tokenizer.get_vocab_size()
        if vocab_size > self.embedding_layer.input_dim:
            new_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=self.embedding_dim)
            new_embedding.build((None,))
            new_embedding.set_weights([
                np.concatenate([
                    self.embedding_layer.get_weights()[0],
                    np.random.normal(scale=0.1, size=(vocab_size - self.embedding_layer.input_dim, self.embedding_dim))
                ], axis=0)
            ])
            self.embedding_layer = new_embedding

    def encode_and_embed(self, tokens):
        ids = self.tokenizer.encode(tokens)
        self.update_embeddings()
        return self.embedding_layer(tf.convert_to_tensor(ids))

class DynamicEmbeddingManager:
    def __init__(self, embedding_dim=16):
        self.embedding_dim = embedding_dim
        self.token_to_id = {}
        self.embedding_matrix = tf.Variable(
            initial_value=tf.random.normal([0, embedding_dim]),  # 초기 크기 0
            trainable=False,
            dtype=tf.float32,
            name="embedding_matrix"
        )

    def get_token_id(self, token):
        """
        토큰이 존재하면 ID 반환, 없으면 새로 추가
        """
        # tf.Tensor이면 numpy로 변환
        if isinstance(token, tf.Tensor):
            token = token.numpy()

        # bytes → str 처리
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        else:
            token = str(token)

        if token not in self.token_to_id:
            new_id = len(self.token_to_id)
            self.token_to_id[token] = new_id
            self._expand_embedding_matrix()  # embedding 행 하나 추가
        return self.token_to_id[token]

    def get_token_ids(self, token_list):
        """
        여러 토큰을 ID 리스트로 반환
        """
        if tf.is_tensor(token_list):
            token_list = token_list.numpy().tolist()
        
        return [self.get_token_id(token) for token in token_list]

    def _expand_embedding_matrix(self):
        """
        새로운 토큰이 생길 때 embedding matrix에 행을 하나 추가
        """
        new_embedding = tf.random.normal([1, self.embedding_dim])
        if self.embedding_matrix.shape[0] == 0:
            self.embedding_matrix = tf.Variable(new_embedding, trainable=False, dtype=tf.float32, name="embedding_matrix")
        else:
            # assign() 대신 embedding_matrix를 새로 생성
            expanded = tf.concat([self.embedding_matrix, new_embedding], axis=0)
            self.embedding_matrix = tf.Variable(expanded, trainable=False, dtype=tf.float32, name="embedding_matrix")

    def get_embeddings(self, token_ids):
        """
        토큰 ID 리스트에 해당하는 임베딩 벡터 반환
        """
        token_ids_tensor = tf.convert_to_tensor(token_ids, dtype=tf.int32)
        return tf.gather(self.embedding_matrix, token_ids_tensor)

    def get_embedding_dim(self):
        return self.embedding_dim

    def get_vocab_size(self):
        return len(self.token_to_id)
    
    def embed_batch(self, token_str_list):
        """
        문자열 토큰 리스트를 받아 임베딩 벡터 배치 반환
        """
        token_ids = self.get_token_ids(token_str_list)
        return self.get_embeddings(token_ids)
