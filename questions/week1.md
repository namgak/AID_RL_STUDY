# 1주차 질문 모음

**각 질문에 대해 일주일 간 각자 고민/질의해보고 여기에 답변을 커밋해주세요 :)**

---

## 1번 질문: `y_pred`와 `reward`의 형태가 궁금합니다.

아래 코드의 `loss_fn` 호출 부분(주석 **#3**)에서 `y_pred`(**#1**)와 `reward`(**#2**)의 값을 비교하는데,  
두 값의 자료형과 형태가 다른 것처럼 보입니다. 어떻게 비교가 가능한 건가요?  
자동으로 변환이 되는 건가요?

```python
def train(env, epochs=5000, learning_rate=1e-2):
    cur_state = torch.Tensor(one_hot(arms, env.get_state()))  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rewards = []

    for i in range(epochs):
        y_pred = model(cur_state)  #1
        av_softmax = softmax(y_pred.data.numpy(), tau=2.0)  
        av_softmax /= av_softmax.sum()  
        choice = np.random.choice(arms, p=av_softmax)  
        cur_reward = env.choose_arm(choice)  

        one_hot_reward = y_pred.data.numpy().copy()  
        one_hot_reward[choice] = cur_reward  
        reward = torch.Tensor(one_hot_reward)  #2
        rewards.append(cur_reward)

        loss = loss_fn(y_pred, reward)  #3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_state = torch.Tensor(one_hot(arms, env.get_state()))
```
## 답:
---

## 2번 질문: `H`의 크기가 100인 이유가 궁금합니다.

원핫 부호화를 통해 10짜리 벡터가 input으로 들어가는 것은 알겠는데 왜 은닉층의 크기가 100이 되나요?

임의로 지정한 것인지 의도가 있는 지 궁금합니다.
```python
import numpy as np
import torch

arms = 10

# N: 배치 사이즈, D_in: 입력 차원, H: 은닉층 크기, D_out: 출력 차원
N, D_in, H, D_out = 1, arms, 100, arms  

# 모델 정의: 입력 → ReLU 은닉층 → 출력 → ReLU
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H), #1
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out), #2
    torch.nn.ReLU(),
)
```

## 답:


