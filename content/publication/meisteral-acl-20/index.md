---
title: "Generalized Entropy Regularization or: There's Nothing Special about Label Smoothing"
date: 2020-07-01
publishDate: 2020-04-06T05:47:36.453048Z
authors: ["Clara Meister", "Elizabeth Salesky", "Ryan Cotterell"]
publication_types: ["1"]
abstract: "Prior work has explored directly regularizing the output distributions of probabilistic models to alleviates peaky (i.e. over-confident) predictions, a common sign of overfitting. This class of techniques, of which label smoothing is one, has a deep mathematical connection to entropy regularization. Despite the consistent success of label smoothing across architectures and data sets in language generation tasks, two problems remain open; (1) there is little understanding of the underlying effects entropy regularizers have on models and (2) the full space of entropy regularization techniques is largely unexplored. We introduce a parametric family of entropy regularizers—which includes label smoothing and the confidence penalty as special cases—and use them to gain a better understanding of the relationship between the entropy of a model’s output distribution and its  performance on language generation tasks. We find that variance in model performance can be explained largely by the resulting entropy of the model’s output distribution rather than by the learning dynamics of the regularizer. Lastly, we find that label smoothing does not allow for sparsity in an output distribution, an undesirable property for language generation models, and therefore advise the use of other entropy regularization methods in its place."
featured: false
publication: "*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*"
publication_short: "ACL"
links:
url_pdf: https://arxiv.org/abs/2005.00820
url_slides: https://drive.google.com/file/d/1C8EPawVTFJrnuUz8RhHR8yFQKhDRy3vZ/view?usp=sharing
---

