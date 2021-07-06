---
title: If beam search is the answer, what was the question?
summary: A deep dive into why beam search generates desirable language from probabilistic language generators.


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
url_pdf: "https://arxiv.org/abs/2010.02650"
url_slides: "https://drive.google.com/file/d/1JfQWAKGLAEBs32fLifZxx58wgMKn3dK9/view?usp=sharing"

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
      <td>University of Amsterdam's computational linguistics seminar</td>
      <td>23.03.21</td>
    </tr>
    <tr>
      <td>NLP with Friends</td>
      <td>03.02.21</td>
    </tr>
    <tr>
      <td>DeepMind's machine translation reading group</td>
      <td>11.12.20</td>
    </tr>
  </tbody>
</table>

### Abstract  
Quite surprisingly, exact maximum a posteriori (MAP) decoding of neural language generators frequently leads to low-quality results. Rather, most state-of-the-art results on language generation tasks are attained using beam search despite its overwhelmingly high search error rate. This implies that the MAP objective alone does not express the properties we desire in text, which merits the question... if beam search is the answer, what was the question? We frame beam search as the exact solution to a different decoding objective in order to gain insights into why high probability under a model alone may not indicate adequacy. We find that beam search enforces uniform information density in text, a property motivated by cognitive science. We suggest a set of decoding objectives that explicitly enforce this property and find that exact decoding with these objectives alleviates the problems encountered when decoding poorly calibrated language generation models. Additionally, we analyze the text produced using various decoding strategies and see that, in our neural machine translation experiments, the extent to which this property is adhered to strongly correlates with BLEU.

