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
=> 현 ARCUSOnline_Embedding, ARCUSOnline_Dynamic에서 trainable=True로 설정해도 업데이트 안됨(그래프에 연결 아되어 있음)
<br>
<br>

| 구분                | EmbeddingManager                                                                         | DynamicEmbeddingManager                                                                  |
| ----------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 핵심 구성             | **Keras Embedding 레이어**                                                                  | **tf.Variable + `tf.gather`**(직접 관리)                                                     |
| 토크나이저             | `DynamicTokenizer` **외부 객체**가 ID 부여                                                      | 내부 `token_to_id` **딕셔너리**가 ID 부여                                                         |
| 확장 방식             | `update_embeddings()`에서 **새 Keras Embedding 레이어**를 만들고 `set_weights`로 **가중치 이어붙임**       | `_expand_embedding_matrix()`에서 `tf.concat` 후 **새 `tf.Variable`로 교체** (행 1개씩 추가)          |
| 그래프 통합            | Keras Layer라 **모델 그래프에 끼워 넣기 쉬움**. `model.fit()`로 **end-to-end 업데이트 가능**(그래프에 연결되어 있을 때) | 레이어가 아니므로 **별도 옵티마이저/테이프**로 `embedding_matrix`를 직접 업데이트해야 함                              |
| 동적 확장 시 옵티마이저 슬롯  | 새 레이어/변수로 **식별자가 바뀌면 슬롯(모멘텀 등) 리셋 가능**                                                   | 행 추가 때마다 **새 Variable 생성** → 슬롯 **자주 리셋** 가능                                             |
| 빈/새 토큰 처리         | `encode()`가 vocab을 늘리고, `update_embeddings()`가 레이어 input\_dim을 **vocab 크기에 맞춰 증가**       | 처음 보는 토큰 **즉시** ID 부여 + **행 1개 생성**                                                      |
| `tf.function` 친화성 | 표준 Keras 경로 → 안정적                                                                        | \*\*사이드 이펙트(ID 추가)\*\*가 있어 `tf.function` 내부에서 호출하면 **재추적/비결정성** 위험. **eager에서 ID 확장** 권장 |
| 에러 패턴             | 대체로 안정. 다만 레이어 교체 시 **옵티마이저 상태/그래프 재바인딩** 고려                                             | **인스턴스 2개**를 섞으면 ID와 가중치 **불일치 → 빈/범위 밖 gather** (MPS에서 assertion)                       |
| 성능/쉬움             | 구현/통합이 **쉬움**, 최적화/훈련 파이프라인과 잘 맞음                                                        | **가볍고 유연**하지만 메모리 재할당/슬롯 리셋 관리 필요                                                        |
| 언제 쓰나             | **모델과 함께 학습**하거나 Keras 친화 파이프라인이면 👍                                                     | **전처리형 임베딩**으로 가볍게 쓰거나, **맞춤형 온라인 튜닝**이 필요하면 👍                                          |

<br>
<br>
핵심 차이 상세
토큰→ID와 임베딩의 소유권
EmbeddingManager: 토큰→ID는 DynamicTokenizer, 임베딩은 Keras Embedding 레이어가 담당. 두 객체를 한 매니저가 감싸는 구조.
DynamicEmbeddingManager: 토큰→ID와 임베딩 둘 다 한 클래스가 직접 관리. 외부 의존 적고 단순.

확장(동적 vocab) 전략
EmbeddingManager: vocab이 커지면 input_dim을 늘린 새 레이어를 만들고, 기존 가중치 + 새 가중치를 concat해 set_weights. 비교적 큰 덩어리로 확장.
DynamicEmbeddingManager: 신규 토큰마다 행 1개를 추가. 구현이 단순하지만 Variable 교체가 잦아 옵티마이저 슬롯이 자주 초기화될 수 있음.
→ 실무에선 **용량(capacity) 선할당 + assign/scatter_update**로 교체 빈도를 줄이는 개선을 권장.

학습 통합 방식
EmbeddingManager: 레이어니까 모델 입력으로 쓰면 자동으로 gradient 흐름. 모델의 train_step에 포함되면 end-to-end로 같이 학습됨.
DynamicEmbeddingManager: 변수 하나라 별도 옵티마이저로 직접 업데이트해야 함.
→ 내가 제안했던 EmbeddingOnlineTunerDEM처럼 보조 손실로 embedding_matrix를 직접 학습하는 패턴이 일반적.

오류/안정성 관점
EmbeddingManager는 레이어가 ID 범위를 보장(= input_dim)해서 빈 gather가 잘 안 남.
DynamicEmbeddingManager는 같은 인스턴스 재사용이 필수. 인코딩용 A, 임베딩용 B처럼 서로 다른 인스턴스를 쓰면 B의 embedding_matrix가 비어 있어 MPS assertion(빈 텐서)로 터짐.
→ 항상 하나의 인스턴스를 전 구간 재사용.
