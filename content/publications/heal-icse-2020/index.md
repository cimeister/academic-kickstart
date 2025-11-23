---
title: "Structure-Invariant Testing for Machine Translation"
date: 2020-08-01
publishDate: 2020-04-06T05:47:36.453048Z
authors: ["Pinjia He", "Clara Meister", "Zhendong Su"]
publication_types: ["1"]
abstract: "Machine translation software has increasingly been integrated into our daily lives. People routinely use machine translation for various applications, such as describing symptoms to a foreign doctor and reading political news in a foreign language. However, due to the complexity and intractability of neural machine translation (NMT) models that power modern machine translation systems, these systems are far from being robust. They can return inferior results that lead to misunderstanding, medical misdiagnoses, threats to personal safety, or political conflicts. Despite its apparent importance, validating the robustness of machine translation is very difficult and has, therefore, been much under-explored.

To tackle this challenge, we introduce structure-invariant testing (SIT), a novel metamorphic testing approach for validating machine translation software. Our key insight is that the translation results of “similar” source sentences should typically exhibit a similar sentence structure. Specifically, SIT (1) generates similar source sentences by substituting one word in a given sentence with semantically similar, syntactically equivalent words, and (2) represents sentence structure by syntax parse trees (obtained via constituency or dependency parsing). To evaluate SIT, we have used it to test Google Translate and Bing Microsoft Translator with 200 source sentences as input, which led to 64 and 70 buggy translations with 69.5% and 70% top-1 accuracy, respectively. The bugs are diverse, including under-translation, over-translation, incorrect modification, word/phrase mistranslation, and unclear logic, none of which could be detected via existing translation quality metrics."
featured: false
publication: "*Proceedings of the 42nd International Conference on Software Engineering*"
publication_short: "ICSE"
links:
url_pdf: https://arxiv.org/abs/1907.08710
---

