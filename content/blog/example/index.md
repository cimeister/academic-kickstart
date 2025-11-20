---
widget: pages
title: 'UnigramLM: The Missing Manual'


summary: Everything you didn't need to know about UnigramLM
abstract: This is my attempt at explaining the math behind the UnigramLM tokenization algorith



# Schedule page publish date (NOT talk date).
publishDate: '2025-11-19T00:00:00Z'

authors: []
tags: []

# Is this a featured talk? (true/false)
featured: true
content:
  # Page type to display. E.g. post, talk, publication...
  page_type: post
# Markdown Slides (optional).
#   Associate this talk with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
view: 1

---

When I first tried to _really_ understand UnigramLM, I did what everyone does: I went to the original 2018 paper. That... didn't get me very far. So then I went to the SentencePiece repo, hoping I could reconstruct the missing pieces from the code. After a brief flashback while staring at the C++ implementation to the terror of my undergraduate CS classes, I bailed on that too. The paper gives a nice high-level story, and the code clearly works in practice, but I couldn't find a single place that actually spells out the full generative model, why the algorithm is mathematically sound, or how all the little "engineering details" (like pruning and vocabulary initialization) fit into that picture.

At that point, I figured to myself, well this is a _unigram_ model. How complicated can it be? I can definitely reason through the logic myself. Turns out that was a tad bit naive. But I'm nothing if not stubborn, so here we are a few months later! 

This post is what I wish I'd had at the start of the endeavor--approachable but rigorous walkthrough of UnigramLM as a probabilistic model, showing why EM is a reasonable tool here, what the posterior over segmentations actually looks like, and how the SentencePiece-style implementation approximates all of this in practice. If you've ever felt that UnigramLM is "clear enough to use, but not clear enough to explain on a whiteboard," my hope is that this takes you the rest of the way to really understanding it, and maybe even extending it. Because at least I think its a pretty cool algorithm that deserves some of BPE's limelight. 