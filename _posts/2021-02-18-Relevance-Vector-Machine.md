---
title: [PRML] Relevance Vector Machine
author: euphoria0-0
date: 2021-02-18 23:56:00 +0800
categories: [AI, Machine Learning]
tags: [Machine Learning, PRML, Bayesian, RVM]
toc: true
math: true
comments: true

---



> *먼저, 이 글은 PRML (CH7) 공부를 바탕으로 작성된 글입니다. 따라서, 간혹 틀리다거나 더 좋은 해석이 있다면 편하게 댓글 부탁드립니다!*



# Relevance Vector Machine

> RVM은 Bayesian SVM이다!  evidence approximation 과정에서 SVM보다 더 sparse해진다.
> 장점: 더 sparse한데도 성능은 좋다. hyper-parameter tuning은 자동적으로 결정된다
> 단점: training 자체는 SVM보다 느리다.

- SVM의 단점
    - 결과가 deterministic
    - binary classification만 잘함
    - hyper-parameter tuning은 validation을 해야함
    - kernel function은 positive definite이어야 하고 training data points를 중심으로 표현돼야 함

## 1. Regression using RVM

RVM은 Bayesian SVM이므로 Bayesian Approach로 SVM을 구하고자 한다.



1. posterior 구하기

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled.png](7%20Sparse%20Kernel%20Machines%20cd030980efa7461fbd6d96449157df79/Untitled.png)

    - posterior 증명

        ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 1.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 1.png)
        
        

2. evidence approximation을 이용한 hyper-parameter 구하기

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 2.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 2.png)

    - likelihood distribution 증명

        ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 3.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 3.png)

    - re-estimated hyper-parameter 증명
    - optimal $$\alpha, \beta$$ 구하는 과정 (evidence 근사 이용)
        1. $$\alpha, \beta$$ 초깃값
        2. (posterior) mean, cov 평가
        3. hyper-parameter 재추정
        4. 수렴까지 2-3. 반복
    
    
    
3. relevance vector와 sparse 의미

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 4.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 4.png)

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 5.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 5.png)

    

4. predictive distribution

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 6.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 6.png)

    - predictive distribution 증명

        ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 7.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 7.png)

- localized basis function의 경우 basis function이 없는 input space의 region에서 예측분산이 작아진다. 이런 경우 RVM은 데이터 도메인 밖에서 extrapolate할수록 예측에 확신을 준다. → *멀리 있는 데이터를 relevance로 선택하게 된다?!*
- RVM의 단점
    - training time이 길다: 하지만 SVM이 hyper-parameter tuning을 위해 validation을 하는 시간을 빼면 생각보다 안 느리고, 더 sparse하므로 빨리 계산할 수 있다.
- RVM은 GP의 한 케이스이다.



## 2. Analysis of Sparsity

1. 예시: 그림을 이용한 직관적 설명

    데이터 2개의 경우,

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 8.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 8.png)

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 9.png](7/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 9.png)

    - $$\varphi$$와 $$\mathbf{t}$$의 방향이 잘 align하면, 이를 relevance vector로 고려하여 모델에 포함시킨다.
    - $$\varphi$$와 $$\mathbf{t}$$의 방향이 잘 align하지 않으면,
        - $$\alpha \rightarrow \infin$$가 되어 해당 항이 0이 되고, 공분산에 대한 $$\varphi$$의 영향이 없어 모델로부터 제거된다.
        - $$\alpha <\infin$$이면(오른쪽 그림) 해당 항에 값이 주어지고 공분산이 커져(퍼져) 데이터에는 낮은 확률이 부여됨. 따라서 $\mathbf{t}$에서의 밀도(확률)값이 낮아짐. 이는 분포가 퍼지고(데이터로부터 멀어짐) → align하는지 안 하는지에 따라 뺄 수 있다.

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled% 10.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 10.png)

2. Sparsity와 Quality

    we make explicit all of the dependence of the marginal likelihood on a particular αi and then determine its stationary points explicitly

    1. posterior의 covariance matrix에서 $\alpha_i$의 기여분을 따로 빼낸다.

        ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 11.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 11.png)

        ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 12.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 12.png)

    2. log likelihood

        ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 13.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 13.png)

    3. $\alpha_i$에 대한 dependence를 포함하는 function

        ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 14.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 14.png)

        ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 15.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 15.png)

        - $s_i$ (sparsity of $\varphi_i$) : basis function이 모델의 다른 basis vector와 overlap되는 정도
        - $q_i$ (quality of $\varphi_i$) : $\mathbf{t}$와 $\mathbf{y}_{-i}$ 간 error와 basis vector $\varphi_i$가 align된 정도
3. Sparsity와 Quality의 상대적인 크기
    1. Stationary points of the marginal likelihood with respect to $\alpha_i$

        ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 16.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 16.png)

        위 식이 0이 될 때는,

        1. $\alpha_i \ge 0$일 때, 
            1. $q_i^2 < s_i$일 때 :  $\alpha_i \rightarrow \infin$ 가 됨
            2. $q_i^2 > s_i$일 때 : 

                ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 17.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 17.png)

        ⇒ 따라서, 이의 상대적 크기가 basis vector가 모델에서 제거되는지 아닌지 결정하게 됨

        → 이는 $\alpha_i$에 대해 closed form 형태의 해가 나타남.

4. Sequential Sparse Bayesian Learning Algorithm

    basis vector가 모델이 포함되는지 아닌지 반복해서 확인

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 18.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 18.png)

5. efficient implementation

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 19.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 19.png)

## 3. RVM for Classification

![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 20.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 20.png)

1. posterior 구하기

    Gaussian approximation 필요 - Laplace Approximation 이용

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 21.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 21.png)

    - gradient, hessian 증명

        ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 22.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 22.png)

2. marginal likelihood

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 23.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 23.png)

    - re-estimated hyper-parameter

        ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 24.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 24.png)

    - $\alpha$ 구하는 과정
        1. $\alpha$ 초기화
        2. initial $\alpha$에 대한 posterior의 Gaussian approximation (marginal likelihood)
        3. $\alpha=\argmax marginal\text{ }likelihood$
        4. 수렴까지 2-3. 반복
- Analysis of Sparsity: classfication case

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 25.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 25.png)

- relevance vector가 decision boundary 쪽에 없다는 것은 $\phi_i(\mathbf{x})$와 $\mathbf{t}$가 잘 align해서 0이 안된 경우이고, 잘 align하지 않으면 0이 되므로 sparse해짐 (잘 align하지 않은 애들은 decision boundary 근처에 있는 애들)

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 26.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 26.png)

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 27.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 27.png)

- multi-class classification

    ![/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 28.png](/assets/img/posts/2021-02-18-Relevance-Vector-Machine/Untitled 28.png)

