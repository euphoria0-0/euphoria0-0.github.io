---
title: Attacks of Embedding Vectors
author: euphoria0-0
date: 2026-01-11 00:00:00 +0800
categories: [Embedding, Security]
tags: [Embedding, Security]
toc: true
math: true
comments: true
---

> Embedding 벡터에 대한 공격에 대해 소개합니다.

## Embedding Vectors란?

AI는 벡터로 표현된 숫자 덩어리를 **언어**로써 세상을 이해합니다.
이를 다양한 형태의 데이터를 고차원 벡터 공간의 점으로 변환한 **임베딩 벡터(Embedding Vector)**라 합니다.

![embedding](/assets/img/posts/2026-01-11/embedding-vector.png)

다양한 데이터를 벡터로 변환하여 AI 모델이 이해하고 처리할 수 있도록 돕습니다.
이는 텍스트, 이미지, 동영상, 오디오 등이 가능합니다.
임베딩 벡터는 RAG를 포함한 문서 검색, 얼굴 인식, 이미지 검색, 추천 시스템 등 다양한 AI 응용 분야에서 핵심적인 역할을 합니다.

최근 연구들에서 임베딩 모델이 서로 다르더라도, 공통의 잠재적 표현이 존재하기 때문에, 지식의 구조가 비슷하게 학습되며, 임베딩 벡터에서 속성의 ranking이 가능함이 보여졌습니다 [1, 2].
그만큼 임베딩은 “정보” 측면에서 강력한 표현력을 가집니다.

## Embedding Vectors 의 위험성

이 압축된 표현 방식은 역설적으로, 정보 벡터로부터 원본 내용을 재구성하는 심각한 위험을 초래할 수 있습니다. 즉, 벡터 임베딩을 사용하는 기존 오픈 API 기반 벡터 데이터베이스를 안전하게 보호하는 것이 점점 더 중요해지고 있습니다.

![embedding](/assets/img/posts/2026-01-11/vulnerability.png)

OWASP, Gartner Research 등에서도 이 문제를 지적하고 있습니다. 즉, 벡터 임베딩을 사용하는 기존 오픈 API 기반 벡터 데이터베이스를 안전하게 보호하는 것이 점점 더 중요해지고 있습니다. [4, 5, 6]

## Text Embedding Inversion

원본 텍스트에 개인정보나 기밀 정보가 있다면 임베딩 벡터 복원으로부터 유출 위험이 있습니다. 이 복원은 임베딩 모델이 무엇인지 몰라도(Black-box setting) 동작할 수 있습니다.

어떻게 작동되는걸까요?

인코더(Encoder)는 임베딩 모델과 최대한 유사하게 학습하고, 
디코더(Decoder)는 인코더의 출력으로부터 텍스트를 재구성합니다.

임베딩 벡터로부터 초기 텍스트를 생성하고 반복적으로 학습 및 수정합니다.

![embedding](/assets/img/posts/2026-01-11/text_paper1.png)

Correction 단계에서 Token, Sequence-level로 반복해서 복원 및 교정합니다.
1. Token-level: 그리디 디코딩(Greedy Decoding)으로 하나의 수정된 텍스트 시퀀스를 생성할 때는 가장 확률이 높은 토큰 하나만을 선택합니다.
2. Sequence-level: Correction 단계 전체를 하나의 단위로 보고 빔 서치(Beam Search)를 적용합니다.

* Beam Search는 그래프 탐색 알고리즘 중 하나로, 가능한 모든 경로를 탐색하는 대신, 매 단계마다 가장 가능성이 높은 상위 후보만을 유지합니다.

## Facial Image Embedding Inversion Attack

얼굴 인식에서 임베딩 벡터의 Similarity Search를 통해 사람을 구분합니다.
이 얼굴 임베딩 벡터로부터도 원본 얼굴 이미지를 복원해낼 수 있고, 얼굴 인식을 악의적으로 통과시키는 공격을 할 수 있습니다.

이미지 복원을 위해 Deconvolution을 이용합니다.
Convolution을 통한 Feature Extraction 과정에서 layer 별로 추상화된 2D 공간적 정보를 Deconvolution에서 local dependency를 고려해 sharp하게 Upsampling합니다.

Loss Function는 아래 함수들의 가중합으로 정의합니다 [7].
1. Image-level
    - MAE (Mean Absolute Error): pixel 단위 L1-norm
    - DSSIM (Dissimilarity Structural Index Metric): Negative SSIM**.

2. Feature-level
    - Perceptual Loss: pretrained VGG-16을 통해 추출한 feature 단위의 L1-norm
    - ID Loss: feature 간 (negative) cosine similarity 와 L1-norm의 합산

** SSIM (Similarity Structural Index Metric)은 이미지의 pixel 단위 평균과 분산을 이용하여 similarity를 계산하며, 여기서 평균과 분산은 사람이 인식하는 이미지 정보인 평균 밝기 및 대비, 이미지 내 패턴/엣지 등의 구조적 정보를 포함하여 정의됩니다.

원 논문에서의 얼굴 임베딩 복원 결과, 일반적인 얼굴인식 시스템의 Threshold 값을 넘어서는 cosine similarity 값을 가지는 것을 볼 수 있습니다.

![embedding](/assets/img/posts/2026-01-11/face_paper2.png)

## 결론

임베딩 벡터는 랜덤한 숫자의 나열이 아닌, 정보가 풍부한 데이터로, 원본 데이터를 거의 완벽하게 복원할 수 있습니다. 따라서 암호화하지 않고 사용하는 기존 솔루션의 경우 정보 유출의 위험이 있습니다.
개인의 민감한 정보를 담은 텍스트 및 얼굴 이미지의 임베딩 벡터는 개인 정보 유출 공격에 굉장히 취약합니다.
AI 기술의 Accuracy, Utility 뿐만 아니라, Security를 위해선 동형암호와 같은 개인정보 보호 기술(Privacy Enhancing Technologies; PETs)의 적용이 시급함을 시사합니다.




## Reference

1. Shmatikov, R. J. C. Z. V., & Morris, J. X. (2025). Harnessing the Universal Geometry of Embeddings. arXiv preprint arXiv:2505.12540.
2. Sonthalia, A., Uselis, A., & Oh, S. J. (2025). On the rankability of visual embeddings. arXiv preprint arXiv:2507.03683.
3. Gartner Research. "Reduce Sensitive Data Leakage Risks From Vector Databases Used in GenAI." 2023. https://www.gartner.com/en/documents/4749631 
4. https://qdrant.tech/documentation/guides/security
5. https://www.mongodb.com/products/capabilities/security
6. John Morris, Volodymyr Kuleshov, Vitaly Shmatikov, and Alexander Rush. 2023. Text Embeddings Reveal (Almost) As Much As Text. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12448–12460, Singapore. ACL.
7. Shahreza, H. O., Hahn, V. K., & Marcel, S. (2024). Vulnerability of state-of-the-art face recognition models to template inversion attack. IEEE Transactions on Information Forensics and Security.

