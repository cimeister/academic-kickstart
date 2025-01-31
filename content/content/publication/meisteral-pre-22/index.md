---
title: "Typical Decoding for Natural Language Generation"
date: 2022-01-01
publishDate: 2022-02-17T11:41:58.327025Z
authors: ["Clara Meister", "Tiago Pimentel", "Gian Wiher", "Ryan Cotterell"]
publication_types: ["2"]
abstract: "Despite achieving incredibly low perplexities on myriad natural language corpora, today's language models still often underperform when used to generate text. This dichotomy has puzzled the language generation community for the last few years. In this work, we posit that the abstraction of natural language as a communication channel (à la Shannon, 1948) can provide new insights into the behaviors of probabilistic language generators, e.g., why high-probability texts can be dull or repetitive. Humans use language as a means of communicating information, and do so in an efficient yet error-minimizing manner, choosing each word in a string with this (perhaps subconscious) goal in mind. We propose that generation from probabilistic models should mimic this behavior. Rather than always choosing words from the high-probability region of the distribution--which have a low Shannon information content--we sample from the set of words with an information content close to its expected value, i.e., close to the conditional entropy of our model. This decision criterion can be realized through a simple and efficient implementation, which we call typical sampling. Automatic and human evaluations show that, in comparison to nucleus and top-k sampling, typical sampling offers competitive performance in terms of quality while consistently reducing the number of degenerate repetitions."
featured: false
publication: "*CoRR*"
links:
url_pdf: papers/meister+al.pre22.pdf
---

