---
title: Causal Discovery - [3] Assumptions
author: euphoria0-0
date: 2024-09-27 00:00:00 +0800
categories: [Causal Learning, Causal Discovery]
tags: [Causal Learning, Causal Discovery]
toc: true
math: true
comments: true
---

> Causal Discovery에 필요한 가정에 대해 소개합니다.

# Assumptions for Causal Discovery

Causal Discovery에 필요한 가정에 대해 소개합니다.

### Acyclicity Assumption

<kbd>Acyclicity</kbd> (비순환성)은 그래프 내 순환성이 없어야 한다는 것을 의미합니다.
우리가 가정하는 그래프가 최소 DAG이므로 DAG의 가정인 Acyclicity 가정이 필수적입니다.
이는 우리가 인과관계를 정의하게 필요합니다.

### Causal Markov Assumption

<kbd>Causal Markov Assumption</kbd> 은 그래프 내에서 관측된 변수는 그래프 내의 부모(parents) 노드에만 의존받으며 자손(descendant) 노드에만 의존성을 부여한다는 것을 의미합니다.

### Faithfulness Assumption

<kbd>Faithfulness Assumption</kbd> 은 그래프 내 연결된 노드는 확률적으로도 의존한다는 것을 의미합니다.

Causal Markov Assumption과 Faithfulness Assumption 두 가정은 서로 역의 관계입니다. 일반적으로 Causal Discovery 방법들은 두 가정을 모두 만족해야 합니다. 

![ASSUMPTION1](/assets/img/posts/2024-09-27-Assumptions/asmp1.png){: width="60%" align="center"}

### Causal Minimality Assumption

<kbd>Causal Minimality Assumption</kbd> 은 불필요한 인과관계는 그래프 내에 존재하지 않아야 함을 내포합니다. SCM의 경우 Y = 0 ⋅ X 와 같은 표기를 허용하지 않는 것을 의미합니다.

### Causal Sufficiency Assumption

<kbd>Causal Sufficiency</kbd> 가정은 그래프 내 모든 변수들의 관측되지 않은 confounder는 존재하지 않아야 함을 의미합니다.



여기서 Causal Discovery에 위의 모든 가정이 필요한 것은 아니며, 최근엔 Acyclicity 등 필수 가정도 완화하는 연구들이 있습니다.

TBD...


## Reference

Pearl, J., Glymour, M., & Jewell, N. P. (2016). Causal inference in statistics: A primer. John Wiley & Sons.

Glymour, C., Zhang, K., & Spirtes, P. (2019). Review of causal discovery methods based on graphical models. *Frontiers in genetics*, *10*, 524.

Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of causal inference: foundations and learning algorithms* (p. 288). The MIT Press.
