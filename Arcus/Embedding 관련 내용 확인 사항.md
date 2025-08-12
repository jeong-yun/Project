1. DynamicEmbeddingManager(freeze) vs EmbeddingManager(Keras, end-to-end)
   - 핵심: vocab 확장 방식과 임베딩을 학습에 연결

- DynamicEmbeddingManager (tf.Variable 기반, 보통 freeze)
  1) 확장 방식: 새 token이 들어오면 embedding_matrix에 행을 1개 더 추가해(concat) 바로 사용. 레이어 재생성 없이 동작.
  2) 시간/메모리: 
     (1) 확장 때마다 O(V×D)[V: vocab 크기(전체 token 수), D: embedding 차원 수] 복사 비용은 있지만, 레이어 재빌드가 없어 오버헤드가 작음  
     (2) trainable = False일 경우, 옵티마이저 슬롯(m,v 등)이 없어 메모리 가볍고 안정적
  4) 학습/안전성:
     (1) 기본은 전처리 embedding → AE의 train_step과 분리 → 임베딩이 시시각각 변하지 않음(희귀/신규 탐지에 유리)  
     (2) 의미 공간을 task에 맞게 '조정'하는 건 별도로 작성하지 않으면 어려움  
        - 모델의 train_step() 안에서 손실에 따라 자동 업데이트되는 경로에 포함되지 않으면, 임베딩 값은 학습 과정에서 변하지 않고 그대로 유지됨  
          * EmbeddingManager(Keras Embedding)는 모델 그래프 안에 포함되므로, train_step()에서 자동으로 업데이트 가능.  
          * DynamicEmbeddingManager는 그래프 밖에서 토큰→벡터를 변환하므로, 임베딩도 학습시키려면 별도로 함수 추가 필요
        <details>
        <summary>코드 보기</summary>
        
        <pre><code class="language-python">with tf.GradientTape() as tape:
            loss = ...
        grads = tape.gradient(loss, [dynamic_embedding_manager.embedding_matrix])
        optimizer.apply_gradients(zip(grads, [dynamic_embedding_manager.embedding_matrix]))</code></pre>
        
        </details>  
  5) 언제 유리?  
     (1) 새 token이 자주 들어오고, 희귀(자주 사용하지 않는)/신규 탐지가 목적일 경우.  
     (2) 실시간 온라인 환경에서 빠르게 안정적으로 돌리고 싶을 경우.

- EmbeddingManager (Keras Embedding, end-to-end 학습 용이) — 기본 값: trainable=True
  1) 확장 방식: update_embeddings() 시 새 레이어를 생성하고 가중치 복사+concat → 레이어 재빌드 오버헤드 발생  
  2) 시간/메모리:  
     (1) 확장 시 일시적으로 메모리 피크(기존+새 레이어 동시 상주 가능)  
     (2) trainable=True면 옵티마이저 슬롯까지 생겨 더 무거워짐  
  3) 학습/표현력:  
     (1) AE의 손실로 임베딩까지 자연스럽게 업데이트(end-to-end)  
     (2) 동의어/변형 등 의미 일반화가 좋아져 오탐 감소 가능 → 다만 embedding이 계속 바뀜 → 희귀/신규 탐지 가능성 낮음  
  4) 언제 유리?  
     (1) 의미 일반화/정밀도가 중요하고, vocab 확장 빈도가 아주 높지 않을 경우.  
     (2) 오프라인 또는 반실시간(버퍼링 가능) 환경

=> 한 줄 요약: 자주 확장 + 안정성(Dynamic+freeze)가 유리, 의미 일반화 + end-to-end면 Keras Embedding이 유리

2. trainable = False vs trainable = True

- trainable = False(freeze)
  1) 임베딩 가중치 고정. 새로운 토큰은 추가만 되고, 기존 토큰 벡터는 변화 X.  
     * 기존 token vector: 고정된 값 유지  
     * 새 token vector: 추가 시 초기화 값으로 세팅 → 이후 그대로 유지  
  2) 장/단점  
     - 장점: 과거 기준/희귀·신규 신호가 일관적(안 흔들림), 메모리/연산 절감.  
     - 단점: 도메인 변화에 적응을 못 함(의미 일반화 개선 한계)

- trainable = True(finetune)
  1) embedding이 AE 손실로 함께 업데이트 → 의미 공간이 적응 → 의미 일반화/동의어 처리 가능  
     * 의미 공간 = 각 token이 vector 공간 상에서 배치된 위치  
     * 적응 = 학습 데이터와 task 특성에 맞게 토큰 vector가 이동  
  2) 장/단점  
     - 장점: 동의어/변형/오탈자 등에도 정확도 ↑, 문맥/의미학습 가능.  
     - 단점: embedding이 변해 희귀/신규 신호가 빨리 흡수될 수 있어 탐지 신뢰도↓, 메모리/연산↑.

=> 한 줄 요약: 목표가 신규/희귀 탐지일 경우 freeze 권장, 목표가 의미 일반화/정밀도 개선일 경우 finetune 고려
