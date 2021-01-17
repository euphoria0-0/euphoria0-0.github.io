---
title: Gaussian Process
author: euphoria0-0
date: 2021-01-17 22:40:00 +0800
categories: [AI, Machine Learning]
tags: [Machine Learning, PRML, Bayesian, Gaussian Process]
toc: true
math: true
comments: true

---



> *먼저, 이 글은 (대부분) PRML (CH6) 공부를 바탕으로 작성된 글입니다. 따라서, 간혹 틀리다거나 더 좋은 해석이 있다면 편하게 댓글 부탁드립니다!*



이 글에서는 Gaussian Process와 Gaussian Process를 Regression과 Classification에 적용하는 Gaussian Process Regression, Gaussian Process Classifier에 대해 다룹니다.



## 1. Gaussian Process

weight가 아닌 function에 대해 prior를 직접 정의한다.

infinite function space에서의 distribution을 고려하는 것은 어렵지만, 실제로 input data point(random variable)에 대한 discrete set에서의 function value만 고려하므로 실제로는 finite space에서 생각할 수 있다.

GP의 예로, kernel regression을 생각해보자.

$$
f(\mathbf{x})=\mathbf{w}^T\phi(\mathbf{x})
$$

$$
p(\mathbf{w})=\mathcal{N}(\mathbf{w}|\mathbf{0},\alpha^{-1}\mathbf{I})
$$
→ $$\mathbf{w}$$에 대한 확률분포를 바탕으로 함수 $$\mathbf{f}$$에 대한 확률분포를 도출할 수 있다.


$$
\mathbf{f}=\mathcal{N}(\mathbf{f}|\mathbf{0},\mathbf{K})
$$


- $$\mathbf{w}\sim\mathcal{N}(\mathbf{w}|\mathbf{0},\alpha^{-1})$$라 가정했고, $$\mathbf{f}$$는 $$\mathbf{w}$$에 대한 선형결합이므로 $$\mathbf{f}$$는 가우시안 분포
- $$\mathbb{E}(\mathbf{f})=\Phi\mathbb{E}(\mathbf{w})=\mathbf{0}$$
- $$\mathrm{Cov}(\mathbf{f})=\mathbb{E}(\mathbf{f}\mathbf{f}^T)=\Phi\mathbb{E}(\mathbf{w}\mathbf{w}^T)\Phi^T=\alpha^{-1}\Phi\Phi^T=\mathbf{K}$$
- $$K_{ij}=k(\mathbf{x}_i,\mathbf{x}_j)=\alpha^{-1}\phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)$$

## 2. Gaussian Process Regression

### 1. weight space view

#### **1. Bayesian formulation about linear regression**

**정의**

- dataset $$\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^n, X=[\mathbf{x}_1^T, \cdots, \mathbf{x}_n^T]$$

- $$\mathbf{f}(\mathbf{x})=\mathbf{x}^T\mathbf{w}$$

- $$\mathbf{y}(\mathbf{x})=\mathbf{f}(\mathbf{x})+\epsilon, \quad \epsilon \sim \mathcal{N}(0,\beta^{-1})$$

- likelihood: $$p(\mathbf{y}|X,\mathbf{w})=\mathcal{N}(\mathbf{y}|X^T\mathbf{w},\beta^{-1})$$

- prior: $$p(\mathbf{w})=\mathcal{N}(\mathbf{0},\mathbf{\Sigma}_p)$$

- posterior: $$p(\mathbf{w}|\mathbf{y},X)=\mathcal{N}(\bar{\mathbf{w}},A^{-1})$$

  - $$p(\mathbf{w}|\mathbf{y},X) = p(\mathbf{y}|X,\mathbf{w})p(\mathbf{w})/p(\mathbf{y}|X)$$

  - $$
    \begin{align*}
    &\log p(\mathbf{w}|\mathbf{y},X) \\
    &\propto\ [-(\mathbf{y}-X^T\mathbf{w})^T\beta\mathbf{I}(\mathbf{y}-X^T\mathbf{w})][-\frac{1}{2}\mathbf{w}^T\mathbf{\Sigma}_p^{-1}\mathbf{w}] \\
    &=-\frac{1}{2}(\mathbf{w}-\bar{\mathbf{w}})^TA(\mathbf{w}-\bar{\mathbf{w}}) \\
    &\textrm{where } \bar{\mathbf{w}}=\beta A^{-1}X\mathbf{y}, A=\beta XX^T+\mathbf{\Sigma}_p^{-1}
    \end{align*}
    $$

    

- predictive distribution

  - $$
    \begin{align*}
    &p(\mathbf{f}_*|\mathbf{x}_*,X,\mathbf{y})\\
    &=\int p(\mathbf{f}_*|X_*,\mathbf{w})p(\mathbf{w}|X,\mathbf{y})d\mathbf{w}\\
    &=\mathcal{N}(\beta X_*^TA^{-1}X\mathbf{y},\mathbf{x}_*^TA^{-1}\mathbf{x}_*)
    \end{align*} \\
    \textrm{where } A=\beta XX^T+\mathbf{\Sigma}_p^{-1}, \mathbf{f}_*=\mathbf{f}_*(\mathbf{x}_*)
    $$

    

#### **2. kernel trick**

- $$\phi: \mathbb{R}^D \rarr \mathbb{R}^N$$ : input space → high dim feature space (N>>D)

- $$\mathbf{f}(\mathbf{x})=\phi(\mathbf{x})^T\mathbf{w}$$
- $$\phi(\mathbf{x})=\left(\phi(\mathbf{x}_1\right) \cdots \phi(\mathbf{x}_n)) \in \mathbb{R}^{N\times n}$$



Then,


$$
\mathbf{f}_*|\mathbf{x}_*,X,\mathbf{y}\sim\mathcal{N}(\beta \phi(\mathbf{x}_*)^TA^{-1}\Phi\mathbf{y},\phi(\mathbf{x}_*)^TA^{-1}\phi(\mathbf{x}_*))
$$


- $$A=\beta\Phi\Phi^T+\mathbf{\Sigma}_p^{-1}, A \in \mathbb{R}^{N\times N}$$
- N>>1, $$A^{-1}$$: computationally incompatible
- $$\mathbf{K}=\Phi^T\Sigma_p\Phi, \phi_*=\phi(\mathbf{x}_*)$$

- $$A\mathbf{\Sigma}_p\Phi=\beta\Phi(\mathbf{K}+\beta^{-1}\mathbf{I})=\mathbf{\Sigma}_p\Phi(\mathbf{K}+\beta^{-1}\mathbf{I})^{-1}$$



Then,

$$
\begin{aligned}\mathbf{f}_*|\mathbf{x}_*,X,\mathbf{y}&\sim\mathcal{N}\left(\beta\phi(\mathbf{x}_*)^TA^{-1}\Phi\mathbf{y}, \phi(\mathbf{x}_*)^TA^{-1}\phi(\mathbf{x}_*)\right) \\ &\sim\mathcal{N}\left(\phi_*\mathbf{\Sigma}_p \Phi(\mathbf{K}+\beta^{-1}\mathbf{I})^{-1}\mathbf{y}, \phi_*^T\mathbf{\Sigma}_p^{-1}\phi_*-\phi_*^T\mathbf{\Sigma}_p\Phi(\mathbf{K}+\beta^{-1}\mathbf{I})^{-1}\Phi^T\mathbf{\Sigma}_p\phi_*\right) \\ &\sim\mathcal{N}\left(k_*(K+\beta^{-1}\mathbf{I})^{-1}\mathbf{y}, k_{**}-k_*(K+\beta^{-1})^{-1}k_* \right) \end{aligned}
$$


### 2. function space view

$$
\mathbf{y}(\mathbf{x})=\mathcal{GP}(m(\mathbf{x}),k(\mathbf{x},\mathbf{x}')+\beta^{-1})
$$



- $$\mathbf{f}=\mathbf{f}(\mathbf{x})=\Phi\mathbf{w}$$
- $$\mathbf{y}=\mathbf{f}+\boldsymbol{\epsilon}, \quad \epsilon_n\sim\mathcal{N}(0,\beta^{-1}),n=1,\cdots,N$$
- $$\mathbf{y}|\mathbf{f}\sim\mathcal{N}(\mathbf{f},\beta^{-1}\mathbf{I}_N)$$
- $$\mathbf{f}\sim\mathcal{N}(\mathbf{0},\mathbf{K}), \quad \mathbf{K}=\alpha^{-1}\Phi\Phi^T$$



#### **1. Inference**

$$
\mathbf{y}\sim\mathcal{N}(\mathbf{y}|\mathbf{0},\mathbf{C}), \quad \mathbf{C}=\mathbf{K}+\beta^{-1}\mathbf{I}_N
$$



(proof)

If $$p(\mathbf{x})=\mathcal{N}(\mu,\Lambda^{-1}), p(\mathbf{y}|\mathbf{x})=\mathcal{N}(A\mathbf{x}+b,L^{-1})$, then $p(\mathbf{y})=\mathcal{N}(A\mu+b,L^{-1}+A\Lambda^{-1}A^T)$$. 

So, 

$$p(\mathbf{y})=\int p(\mathbf{y}|\mathbf{f})p(\mathbf{f})d\mathbf{y}=\mathcal{N}(\mathbf{0},\beta^{-1}\mathbf{I}_N+\mathbf{K})$$



**2. Prediction**

predictive value(vector) $$\mathbf{f}_*$$ about new input $$\mathbf{x}_*$$

$$
\begin{pmatrix} \mathbf{y} \\ \mathbf{y}_* \end{pmatrix} \sim \left( \begin{pmatrix} \mathbf{0} \\ 0 \end{pmatrix}, \begin{pmatrix} \mathbf{K}+\beta^{-1}\mathbf{I}_N & \mathbf{k}_* \\ \mathbf{k}_*^T & \mathbf{k}_{**}+\beta^{-1} \end{pmatrix} \right)
$$
where $$\mathbf{k}_*=\mathbf{k}(\mathbf{x}_n,\mathbf{x}_*), \mathbf{k}_{**}=\mathbf{k}(\mathbf{x}_*,\mathbf{x}_*)$$


$$
\mathbf{y}_*|\mathbf{y}\sim\mathcal{N}\left(\mathbf{k}^T(\mathbf{K}+\beta^{-1}\mathbf{I}_N)^{-1}\mathbf{y}, \mathbf{k}_{**}+\beta^{-1}-\mathbf{k}_*^T(\mathbf{K}+\beta^{-1}\mathbf{I}_N)^{-1}\mathbf{k}_*\right)
$$


- (proof)
  - Using lemma $$p(\mathbf{x}_a|\mathbf{x}_b)=\mathcal{N}\left(\boldsymbol{\mu}_a+\mathbf{\Sigma}_{ab}\mathbf{\Sigma}_{bb}^{-1}(\mathbf{x}_b-\boldsymbol{\mu}_b), \mathbf{\Sigma}_{aa}-\mathbf{\Sigma}_{ab}\mathbf{\Sigma}_{bb}^{-1}\mathbf{\Sigma}_{ba}\right)$$,
  - $$\boldsymbol{\mu}_{\mathbf{y}_*|\mathbf{y}}=0+k_*^T(K+\beta^{-1}I)^{-1}(\mathbf{y}-0)=k_*C^{-1}\mathbf{y}$$
  - $$\Sigma_{\mathbf{y}_*|\mathbf{y}}=k_{**}+\beta^{-1}-k_*^T(K+\beta^{-1}I_N)^{-1}k_
    *=k_{**}+\beta^{-1}-k_*^TC^{-1}k_*$$



- $$\mathbf{C}$$는 positive definite이어야 한다!

  - $$K$$의 eigen value ≥0 ⇒ $$k(x_i,x_j)$$가 모든 $$x_i,x_j$$에 대해 positive definite이게 됨

- predictive mean

  $$\mathbf{k}_*^T\mathbf{C}^{-1}\mathbf{y}=\sum_{n=1}^Na_n\mathbf{k}(\mathbf{x}_n,\mathbf{x}_*), \quad a_n=[\mathbf{C}^{-1}\mathbf{y}]_n$$

  - $$M<<N$$일 때 GP가 효율적이다.



### 3. Hyper-parameter

MLE에서 likelihood $$p(\mathbf{y}|\theta)$$를 계산 → conjugate gradients와 같은 방법으로

- $$\log p(\mathbf{y}|\theta)=\frac{1}{2}\log|\mathbf{C}|-\frac{1}{2}\mathbf{y}^T\mathbf{C}^{-1}\mathbf{y}-\frac{N}{2}\log(2\pi)$$

- $$\frac{\partial}{\partial\theta_i}\log p(\mathbf{y}|\theta)=-\frac{1}{2}Tr\left( \mathbf{C}^{-1} \frac{\partial C}{\partial\theta_i}\right)-\frac{1}{2}\mathbf{y}^T(-\mathbf{C}^{-1}\frac{\partial C}{\partial\theta_i}\mathbf{C}^{-1})\mathbf{y}$$

- 위는 non-convex이므로 계산 어려움

- fully Bayesian으로 $$\theta$$에 prior 주는 계산은 어려움



### 4. ARD: Automatic Relevance Determination





## 3. Gaussian Process Classifier









###### Reference

1. Bishop, C. M. (2006). *Pattern recognition and machine learning*. springer.
2. Rasmussen, C. E. (2003, February). Gaussian processes in machine learning. In *Summer School on Machine Learning* (pp. 63-71). Springer, Berlin, Heidelberg.
3. 최성준 교수님, edwithh, Bayesian Deep Learning