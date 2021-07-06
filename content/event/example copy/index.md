---
title: Evidence for the uniform information density hypothesis in modern NLP models
summary: Our recent work explores how the uniform information density hypothesis can be operationalized for use in models of natural language processing.


# Talk start and end times.
#   End time can optionally be hidden by prefixing the line with `#`.
# Schedule page publish date (NOT talk date).
publishDate: "2017-01-01T00:00:00Z"
showthedate: false
date: "2017-01-02"

authors: []
tags: []

# Is this a featured talk? (true/false)
featured: false

image:
  caption: ''
  focal_point: Left

links:
- icon: twitter
  icon_pack: fab
  name: Follow
  url: https://twitter.com/clara__meister
url_pdf: "https://arxiv.org/abs/2105.07144"
url_slides: 

# Markdown Slides (optional).
#   Associate this talk with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
# slides: example

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.

---
<table class="table">
  <head>
    <base target="_blank">
  </head>
  <thead>
    <tr>
      <th scope="col" style='white-space:nowrap'>Location</th>
      <th scope="col" style='white-space:nowrap'>Date&emsp;&emsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Berkeley's NLP Seminar</td>
      <td>14.07.21</td>
    </tr>
    <tr>
      <td>MIT's Computational Psycholinguistics Lab</td>
      <td>16.06.21</td>
    </tr>
  </tbody>
</table>

### Abstract  
In this talk, I will review two recent works that have operationalized the uniform information density (UID) hypothesis for use in models of natural language processing. In machine translation, it has been frequently observed that texts assigned high probability (i.e., low surprisal) are not necessarily what humans perceive to be high quality language. Alternatively, text decoded using beam search, a popular heuristic decoding method, often scores well in terms of both qualitative and automatic evaluation metrics, such as BLEU. We show that beam search can be framed as a UID-enforcing decoding objective and that there exists a strong relationship between BLEU and the extent to which UID is adhered to in natural language text. 
In a follow up work, we explore the effects of directly incorporating an operationalization of UID into a language model's training objective. Specifically, we augment the canonical MLE objective with a regularizer that encodes UID. In experiments on ten languages spanning five language families, we find that using UID regularization consistently improves perplexity in language models, having a larger effect when training data is limited. Moreover, via an analysis of generated sequences, we find that UID-regularized language models have other desirable properties, e.g., they generate text that is more lexically diverse.

