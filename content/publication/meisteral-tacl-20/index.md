---
title: "Best-First Beam Search"
date: 2020-08-01
publishDate: 2020-05-06T05:43:04.543051Z
authors: ["Clara Meister", "Tim Vieira", "Ryan Cotterell"]
publication_types: ["1"]
abstract: "Decoding for many NLP tasks requires a heuristic algorithm for approximating exact search since the full search space is often intractable if not simply too large to traverse efficiently. The default algorithm for this job is beam search--a pruned version of breadth-first search--which in practice, returns better results than exact inference due to beneficial search bias. In this work, we show that standard beam search is a computationally inefficient choice for many decoding tasks; specifically, when the scoring function is a monotonic function in sequence length, other search algorithms can be used to reduce the number of calls to the scoring function (e.g., a neural network), which is often the bottleneck computation. We propose best-first beam search, an algorithm that provably returns the same set of results as standard beam search, albeit in the minimum number of scoring function calls to guarantee optimality (modulo beam size).  We show that best-first beam search can be used with length normalization and mutual information decoding, among other rescoring functions.  Lastly, we propose a memory-reduced variant of best-first beam search, which has a similar search bias in terms of downstream performance, but runs in a fraction of the time."
featured: false
publication: "*Transactions of the Association for Computational Linguistics*"
publication_short: "TACL"
links:
url_pdf: https://arxiv.org/abs/2007.03909
url_slides: https://drive.google.com/file/d/1RJoeh4vpAkGyGqUZtqTHfwJys-wxl8z_/view?usp=sharing
---

