# arcusonline_embedding.py
# MacOS 환경으로 인한 float64에서 float32로 변경
# 리팩터 요약:
# - 토큰 → 동적 vocab → 임베딩(keras Embedding) → 윈도우 고정차원 피처 → 코어 이상치 모델(RSRAE/RAPP/DAGMM)
# - 온라인 학습: 배치가 들어올 때마다 코어 모델 미세학습, 드리프트 시 새 모델 생성 및 병합 유지
# - 동적 임베딩: 새 토큰 유입 시 vocab 확장 & 임베딩 매트릭스 확장 (가중치 보존)
# - 핵심 변경: ARCUS._build_window_vector_median()가 EmbeddingManager.embedding_layer를 사용하도록 변경
# - 주의: 코어 모델의 train_step이 내부에서 최적화를 수행하므로, 임베딩 가중치(embedding_layer)는 
#         기본적으로 코어 모델의 그래프와 분리되어 업데이트되지 않습니다.
#         (본 예시는 '전처리 임베딩 + 온라인 학습'을 안전하게 구성하는 리팩터입니다.)
#         임베딩까지 end-to-end 학습하려면, 코어 모델의 train_step을 개방하거나,
#         아래의 TODO 섹션처럼 별도의 임베딩 업데이트 루프를 추가해야 합니다.

import tensorflow as tf
import numpy as np
from sklearn import metrics
from CKA import linear_CKA
from model.model_utils import ModelGenerator

from embedding_manager import DynamicEmbeddingManager, EmbeddingManager
from collections import Counter

class ARCUSOnline_Embedding:
    def __init__(self, args):
        self.seed = args.seed
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        # ARCUS 설정
        self._model_type = args.model_type
        self._inf_type   = args.inf_type
        self._itr_num    = int(args.batch_size/args.min_batch_size)

        self._batch_size = args.batch_size
        self._min_batch_size = args.min_batch_size

        self._init_epoch = args.init_epoch
        self._intm_epoch = args.intm_epoch

        self._reliability_thred = args.reliability_thred
        self._similarity_thred  = args.similarity_thred

        # 동적 임베딩 설정
        self._embedding_dim = args.embedding_dim
        self.use_trainable_embedding_layer = True  # keras Embedding 사용 여부

        # 윈도우 통계 추적(선택)
        self.token_counts = {}
        self.seen_tokens = set()
        self.rare_min_count = 5

        self._model_generator = ModelGenerator(args)

        # 2가지 임베딩 매니저 제공
        # (1) keras Embedding 기반: 가중치 보존하며 확장, 레이어 자체는 trainable=True
        self.embedding_manager_layer = EmbeddingManager(embedding_dim=args.embedding_dim)
        # (2) tf.Variable 기반: 간단/가벼움. 필요 시 사용
        self.dynamic_embedding_manager = DynamicEmbeddingManager(embedding_dim=args.embedding_dim)

        # 모델 풀
        self.model_pool = []

    # -----------------------------
    # 임베딩 전처리 → 윈도우 고정차원 피처 만들기
    # -----------------------------
    def _tokens_to_ids(self, token_list):
        # keras Embedding 레이어 기반 vocab 확장
        ids = self.embedding_manager_layer.tokenizer.encode(token_list)
        self.embedding_manager_layer.update_embeddings()  # vocab 확장 시 가중치 보존
        return ids

    def _window_features_median(self, token_str_list):
        """
        Embedding(keras) → 중앙값 임베딩 + 거리 통계 + 간단 빈도 통계
        출력: (D+6,) 텐서
        """
        if token_str_list is None or len(token_str_list) == 0:
            D = self._embedding_dim
            zeros = tf.zeros((D + 6,), dtype=tf.float32)
            return zeros

        # 1) 토큰 → ID → 임베딩(keras)
        ids = self._tokens_to_ids(token_str_list)                # Python list[int]
        ids_tf = tf.convert_to_tensor(ids, dtype=tf.int32)       # (L,)
        E = self.embedding_manager_layer.embedding_layer(ids_tf) # (L, D) float32, trainable=True
        E_np = E.numpy()                                         # 통계용 (median/iqr)

        # 2) 중앙값 임베딩
        med = np.median(E_np, axis=0)                # (D,)
        # 3) 중앙값 기준 거리 통계
        dists = np.linalg.norm(E_np - med, axis=1)   # (L,)
        med_dist  = float(np.median(dists))
        iqr_dist  = float(np.percentile(dists, 75) - np.percentile(dists, 25))
        max_dist  = float(np.max(dists))
        mean_dist = float(np.mean(dists))

        # 4) 윈도우 내부 빈도 통계
        L = max(1, len(token_str_list))
        cnt = Counter(token_str_list)
        unique_ratio = len(cnt) / L
        top1_ratio   = max(cnt.values()) / L

        median_embed = tf.convert_to_tensor(med, dtype=tf.float32)  # (D,)
        scalars = tf.convert_to_tensor(
            [med_dist, iqr_dist, max_dist, mean_dist, unique_ratio, top1_ratio],
            dtype=tf.float32
        )
        window_vec = tf.concat([median_embed, scalars], axis=0)      # (D+6,)
        return window_vec

    def _standardize_scores(self, score):
        score = np.asarray(score, dtype=np.float32)
        mean = float(np.mean(score))
        std  = float(np.std(score))
        if std < 1e-6:
            return score
        return (score - mean) / std

    def _merge_models(self, model1, model2):
        # 가중 병합(원본 ARCUS 그대로 유지)
        num_batch_sum = model1.num_batch + model2.num_batch
        w1 = model1.num_batch / num_batch_sum
        w2 = model2.num_batch / num_batch_sum

        # encoder 병합
        for layer_idx in range(len(model2.encoder)):
            l_base = model1.encoder[layer_idx]
            l_target = model2.encoder[layer_idx]
            if l_base.name[:5] == 'layer':
                new_weight = (l_base.weights[0] * w1 + l_target.weights[0] * w2)
                new_bias = (l_base.weights[1] * w1 + l_target.weights[1] * w2)
                l_target.set_weights([new_weight, new_bias])
            elif l_base.name[:2] == 'bn':
                new_gamma = (l_base.weights[0] * w1 + l_target.weights[0] * w2)
                new_beta  = (l_base.weights[1] * w1 + l_target.weights[1] * w2)
                new_mm    = (l_base.weights[2] * w1 + l_target.weights[2] * w2)
                new_mv    = (l_base.weights[3] * w1 + l_target.weights[3] * w2)
                l_target.set_weights([new_gamma, new_beta, new_mm, new_mv])

        # decoder 병합
        for layer_idx in range(len(model2.decoder)):
            l_base = model1.decoder[layer_idx]
            l_target = model2.decoder[layer_idx]
            if l_base.name[:5] == 'layer':
                new_weight = (l_base.weights[0] * w1 + l_target.weights[0] * w2)
                new_bias = (l_base.weights[1] * w1 + l_target.weights[1] * w2)
                l_target.set_weights([new_weight, new_bias])
            elif l_base.name[:2] == 'bn':
                new_gamma = (l_base.weights[0] * w1 + l_target.weights[0] * w2)
                new_beta  = (l_base.weights[1] * w1 + l_target.weights[1] * w2)
                new_mm    = (l_base.weights[2] * w1 + l_target.weights[2] * w2)
                new_mv    = (l_base.weights[3] * w1 + l_target.weights[3] * w2)
                l_target.set_weights([new_gamma, new_beta, new_mm, new_mv])

        if self._model_type == 'RSRAE':
            model2.A = (model1.A * w1 + model2.A * w2)

        model2.num_batch = num_batch_sum
        return model2

    def _reduce_models_last(self, x_inp, epochs):
        latents = []
        for m in self.model_pool:
            z = m.get_latent(x_inp)
            latents.append(z.numpy())

        max_CKA = 0.0
        max_Idx1 = None
        max_Idx2 = len(latents) - 1
        for idx1 in range(len(latents)-1):
            CKA = linear_CKA(latents[idx1], latents[max_Idx2])
            if CKA > max_CKA:
                max_CKA = CKA
                max_Idx1 = idx1

        if max_Idx1 is not None and max_CKA >= self._similarity_thred:
            self.model_pool[max_Idx2] = self._merge_models(self.model_pool[max_Idx1], self.model_pool[max_Idx2])
            self._train_model(self.model_pool[max_Idx2], x_inp, epochs)
            self.model_pool.remove(self.model_pool[max_Idx1])
            if len(self.model_pool) > 1:
                self._reduce_models_last(x_inp, epochs)

    def _train_model(self, model, x_inp, epochs):
        num_samples = x_inp.shape[0]
        batch_size = self._min_batch_size
        num_iters = max(1, num_samples // batch_size)

        tmp_losses = []
        for _ in range(epochs):
            for _ in range(num_iters):
                min_batch_x_inp = tf.random.shuffle(x_inp)[:batch_size]
                if tf.reduce_any(tf.math.is_nan(min_batch_x_inp)):
                    continue
                if tf.reduce_all(tf.equal(min_batch_x_inp, 0.0)):
                    continue
                loss = model.train_step(min_batch_x_inp)
                if tf.reduce_any(tf.math.is_nan(loss)):
                    continue
                tmp_losses.append(float(loss.numpy()))
        temp_scores = model.inference_step(x_inp)
        model.last_mean_score = float(np.mean(temp_scores))
        model.last_max_score = float(np.max(temp_scores))
        model.last_min_score = float(np.min(temp_scores))
        model.num_batch = getattr(model, "num_batch", 0) + 1
        return tmp_losses


    def simulator1(self, loader):
        """Unsupervised online anomaly detection (샘플 단위, 동적 임베딩 + 온라인 미세학습)"""
        self.model_pool = []

        drift_hist, losses = [], []
        all_scores, all_batches, all_vector = [], [], []

        first = True
        try:
            for step, (x_inp, y_inp) in enumerate(loader.batch(self._batch_size)):
                # 1) (B, ...) → (B, F) 윈도우 피처로 변환
                batch_window_vecs = []
                x_iter = tf.unstack(x_inp) if tf.is_tensor(x_inp) else list(x_inp)

                for xi in x_iter:
                    # xi → 토큰 문자열 리스트로 변환
                    if tf.is_tensor(xi):
                        flat = tf.reshape(xi, [-1])
                        tokens = []
                        for t in flat:
                            v = t.numpy()
                            if isinstance(v, bytes):
                                tokens.append(v.decode("utf-8", errors="ignore"))
                            else:
                                tokens.append(str(v))
                    else:
                        tokens = [str(t) for t in np.array(xi).reshape(-1)]

                    window_vec = self._window_features_median(tokens)  # (F,)
                    batch_window_vecs.append(window_vec)

                x_batch = tf.stack(batch_window_vecs, axis=0)         # (B, F)
                x_batch = tf.cast(x_batch, tf.float32)

                B = int(x_batch.shape[0])
                F = int(x_batch.shape[1])

                all_batches.append(np.median(x_batch.numpy(), axis=1))
                all_vector.extend([x_batch[i].numpy() for i in range(B)])

                # 2) 최초 1회: 모델 초기화 + 초기학습
                if first:
                    initial_model = self._model_generator.init_model(input_dim=F)
                    self.model_pool = [initial_model]
                    curr_model = initial_model
                    losses += self._train_model(initial_model, x_batch, self._init_epoch)
                    first = False
                else:
                    curr_model = self.model_pool[0]

                # 3) 추론 & 신뢰도 계산
                if self._inf_type == "INC":
                    final_scores = self.model_pool[0].inference_step(x_batch)
                else:
                    scores, model_reliabilities = [], []
                    for m in self.model_pool:
                        s = m.inference_step(x_batch)
                        scores.append(s)
                        curr_mean = float(np.mean(s))
                        curr_max  = float(np.max(s))
                        curr_min  = float(np.min(s))

                        last_mean = getattr(m, "last_mean_score", curr_mean)
                        last_max  = getattr(m, "last_max_score",  curr_max)
                        last_min  = getattr(m, "last_min_score",  curr_min)

                        min_score = min(curr_min, last_min)
                        max_score = max(curr_max, last_max)
                        gap       = abs(curr_mean - last_mean)
                        denom = (2 / max(1, B)) * (max_score - min_score) ** 2
                        if denom < 1e-8 or np.isnan(denom):
                            denom = 1e-8
                        reliability = float(np.exp(-2 * (gap ** 2) / denom))
                        model_reliabilities.append(reliability)

                    curr_model_idx = int(np.argmax(model_reliabilities))
                    curr_model = self.model_pool[curr_model_idx]

                    weighted_scores = []
                    for idx in range(len(self.model_pool)):
                        w = model_reliabilities[idx]
                        standardized = self._standardize_scores(scores[idx])
                        weighted_scores.append(standardized * w)
                    final_scores = tf.reduce_sum(tf.stack(weighted_scores, axis=0), axis=0)

                all_scores += list(final_scores.numpy())

                # 4) 드리프트 판단
                if self._inf_type == "INC":
                    drift = False
                else:
                    pool_reliability = 1 - np.prod([1 - p for p in model_reliabilities])
                    drift = pool_reliability < self._reliability_thred

                # 5) 모델 적응
                if drift:
                    drift_hist.append(step)
                    new_model = self._model_generator.init_model(input_dim=F)
                    losses += self._train_model(new_model, x_batch, self._init_epoch)
                    self.model_pool.append(new_model)
                    self._reduce_models_last(x_batch, 1)
                else:
                    losses += self._train_model(curr_model, x_batch, self._intm_epoch)

        except Exception as e:
            print("At seed:", self.seed, "Error:", e)
            return False, None, None, None, None, None, None

        model_count = len(self.model_pool)
        trained_models = self.model_pool
        return True, all_scores, model_count, drift_hist, trained_models, all_batches, all_vector


# -----------------------------
# TODO: 임베딩 가중치까지 온라인으로 업데이트하기 (선택)
# -----------------------------
# 코어 모델의 train_step이 내부에서 옵티마이저를 적용하는 구조라, 임베딩 레이어까지 
# 같은 손실로 end-to-end 업데이트하려면 다음 중 하나가 필요합니다.
#  1) 코어 모델들이 forward(loss 계산만)과 apply_gradients를 분리하여 제공
#  2) ARCUS 층에서 tf.GradientTape으로 코어 모델 forward를 감싼 뒤,
#     tape.gradient(loss, model_vars + [embedding weights])로 한 번에 업데이트
#  3) 또는 임베딩 전용 보조목표(예: 윈도우 내 임베딩 분산/노이즈 안정화 등)를 정의하여 별도 옵티마이저로 미세 조정
# 아래 스켈레톤은 (3)번의 간단한 예시입니다.

class EmbeddingOnlineTuner:
    """임베딩 레이어를 간단한 보조 손실로 온라인 미세조정하는 예시."""
    def __init__(self, embedding_manager_layer: EmbeddingManager, lr=1e-3):
        self.manager = embedding_manager_layer
        self.opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def step(self, ids):
        with tf.GradientTape() as tape:
            emb = self.manager.embedding_layer(ids)  # (L, D)
            # 보조 손실: L2 정규화 + 윈도우 내 분산을 적당히 제어
            var_loss = tf.reduce_mean(tf.math.reduce_variance(emb, axis=0))
            l2_loss  = tf.reduce_mean(tf.square(emb))
            loss = 0.1 * var_loss + 1e-4 * l2_loss
        grads = tape.gradient(loss, self.manager.embedding_layer.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.manager.embedding_layer.trainable_variables))
        return loss
