---
title: "Sparsity Tricks"
tags:
  - model-tricks
  - sparsity
  - optimization
date: 2026-06-11
summary: "Notes on sparse linear weights and sparsity techniques in neural networks"
---

# Sparse Linear Weights

### Linear Layer

Instead of this 

$Y = W.X + B$

Just use a low rank $W$

$Y = (U.V).X + B$

Where $W = dim(m , n)$ and $U=dim(m , t) V=dim(t , n) , t < m,n$

### Shared input multiple projections



