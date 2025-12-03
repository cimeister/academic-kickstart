
**TL;DR**:  This post is my attempt to write down the UnigramLM tokenization algorithm cleanly and explicitly because, well, I still haven't found such a derivation and I think understanding the theory behind the method could help us make it better. I'll formalize the generative model on which the algorithm is based, derive the EM updates, explain why pruning is needed (and how it's done), and point out the
spots where the practical implementation defined by the SentencePiece library diverges from the pretty mathematical models. I hope this post provides a new lens through which to look at the UnigramLM tokenization algorithim while pointing out some interesting potential extensions/revisions to the current implementation. 

### Intro and origins of this blog post
*(feel free to [skip](#sec:background) this section)*

These days, tokenization is basically synonymous with Byte-pair Encoding (BPE). If you ask someone "do you know how tokenization works?", there's a decent chance you'll get an answer like: "Yeah yeah, I know BPE."  But tokenization != BPE. There are numerous (arguably better motivated) algorithms one could use for segmenting text into tokens. This post focuses on UnigramLM (the SentencePiece "unigram" model), which is a pretty far departure from the BPE approach... 

#### Why look at UnigramLM now (and not just "make BPE better")?
Recent work keeps showing that tokenizers themselves can induce [unfairness](https://arxiv.org/abs/2305.15425) and [uneven performance](https://arxiv.org/abs/2305.17179) across languages, dialects, and writing systems. A lot of the community response has (reasonably!) focused on patching BPE: adding constraints, regularizers, or parity-aware merges. Those are valuable, but there's a risk in treating "tokenization = BPE + tweaks" as the whole design space. UnigramLM is a widely deployed alternative (T5, XLNet), and it comes from a fundamentally different modeling viewpoint. Instead of greedily merging pairs, it says: "let's uncover latent tokens and treat tokenization like inference." At least to me, that framing feels a lot more linguistically sane (or, at minimum, less like we're playing subword Tetris). Taking that viewpoint seriously could open different and maybe cleaner directions for addressing tokenizer-induced unfairness---not by iterating on one algorithm forever, but by re-examining the assumptions we bake into tokenization in the first place.

#### Why this blog post

With the above motivation in mind, I figured I should actually understand the algorithm. 
So I did what everyone does: I went to the [original 2018 paper](https://aclanthology.org/P18-1007/). That... didn't get me very far. So then I went to the SentencePiece repo, hoping I could reconstruct the missing pieces from the code. After a brief flashback to the terror of my undergraduate CS classes while staring at the C++ implementation, I bailed on that approach too. Then I thought maybe the missing explanation was hiding in the HuggingFace documentation. But let's just say that rabbit hole ended like this:


> *The HuggingFace documentation* [on UnigramLM] *describes a
> tokeniser that doesn't exist. It should not be relied on as an
> explanation for UnigramLM, because it doesn't even come close.*  
> –Claude

The original UnigramLM paper gives a nice high-level story, and the code clearly works in practice, but I couldn't find a single place that actually spells out the full generative model, why the algorithm is mathematically sound, or how all the little "engineering details" (like pruning and vocabulary initialization) fit into that picture. This post is my attempt to provide an approachable but rigorous walkthrough of UnigramLM as a probabilistic model, showing why EM is a reasonable tool here, what the posterior over segmentations actually looks like, and how the SentencePiece-style implementation approximates/diverges from all of this in practice. If you've ever felt that UnigramLM is "clear enough to use, but not clear enough to explain on a whiteboard," my hope is that this takes you the rest of the way to really understanding it, and maybe even extending it. Because at least I think its a pretty cool algorithm that deserves some of BPE's limelight. 




## Tokenization Background and Notation {#sec:background}

So that we're on the same page, let's start with a formal definition of tokenization. 


## What you came here for: UnigramLM

The UnigramLM tokenization algorithm (Kudo 2018) takes a
probabilistic-modeling approach to string tokenization. It defines an
${h}$, together with an algorithm for learning its parameters, by
treating tokenization as inference in a latent-variable generative model
over strings---in particular, a unigram generative model.

**Few Sentence Description of UnigramLM**: UnigramLM is basically what it sounds like: a unigram language model. The only parameters of the tokenization scheme are a unigram probability distribution. When learning the tokenizer, we learn both the vocabulary and piece probabilities of this unigram model that (approximately) maximize corpus log-likelihood. At inference time, given a string, UnigramLM chooses the segmentation (sequence of pieces) that has the highest probability under this learned unigram model. In contrast to BPE's greedy merge story, UnigramLM's behavior is really "whichever segmentation makes the whole corpus most probable under this unigram model wins."


#### Concise Pseudocode {#sec:pseudocode}


    Algorithm UnigramLM-Train(C, V_target_size, V0, phi0, k_n):
        V   <- V0
        phi <- phi0

        while |V| > V_target_size:

            # ---- E-step ----
            hat_c[v] <- 0 for all v in V
            for each string s in C:
                lattice <- build_lattice(s, V)
                tilde_c <- forward_backward_expected_counts(lattice, phi)
                for each v in V:
                    hat_c[v] <- hat_c[v] + tilde_c[v]

            # ---- M-step ----
            Z <- sum_{v in V} hat_c[v]
            for each v in V:
                phi[v] <- hat_c[v] / Z

            # ---- Pruning (approx loss) ----
            for each v in V:
                v_alt <- viterbi_best_segmentation(g(v), V \ {v}, phi)
                alt_prob <- product_{t in v_alt} phi[t]
                Lhat[v] <- hat_c[v] * ( log(phi[v]) - log(alt_prob) )

            V <- V \ bottom_k_tokens_by(Lhat, k_n)
            phi <- renormalize(phi over V)

        return V, phi

Third, 
SentencePiece does not use the plain MLE
M-step update from
Eq. (14). Instead, it adopts a Variational Bayesian
approach with a Dirichlet prior, replacing expected counts with their
[digamma-transformed counterparts](https://github.com/google/sentencepiece/blob/336900241c4943ae1e5f844b18292f532b3a21c7/src/unigram_model_trainer.cc#L390):
${\phi^{(n+1)}_{v}}\propto\exp(\Psi(\widehat{c}_{{v}}({\mathcal{C}};{{\boldsymbol{\phi}}^{(n)}})+\alpha_{v}))$.
While it might not seem like a large change to the original update rule, this choice is implicitly adding a
prior belief about the the number of counts we should observe for each token. 
Explicitly, we're now calculating the geometric mean of a
posterior Dirichlet distribution, where we're added in the belief token ${v}$ will be observed $\alpha_{v}$  times. 
Notably, SentencePiece uses an improper Haldane prior ($\alpha_{v}= 0$
for all ${v}\in{\mathcal{V}}$). This choice essentially has the opposite
effect of performing standard additive smoothing: it's always the case
that $\exp(\psi(x)) < x$, however, for small $x$ (rare tokens), the
relative "discount" is significantly larger. It thus acts as a
regularizer that disproportionately penalizes tokens with low expected
counts, sending their assigned probability mass closer to zero.  This is done on top of a [pre-pruning step](https://github.com/google/sentencepiece/blob/336900241c4943ae1e5f844b18292f532b3a21c7/src/unigram_model_trainer.cc#L381) for tokens whose expected counts are below a certain threshold. 

https://github.com/google/sentencepiece/blob/53de76561cfc149d3c01037f0595669ad32a5e7c/doc/options.mds




## Conclusions

Tokenization shouldn't be just a monolithic preprocessing step you fix once and forget; it quietly defines what your model even sees as input, and can have a huge effect on the behavior and fairness of the systems trained on top of it. If we take that seriously, we should treat tokenization as a full-blown modeling choice and explore the whole design space: priors (e.g., over length), supports (which segmentations are even allowed by pretokenization choices), and inference rules (Viterbi vs sampling vs marginalization). UnigramLM occupies just one corner of that space, but understanding it clearly is a step toward thinking about tokenizers as models we can design and question, not just as default settings we inherit.

#### _Acknowledgments_

As with pretty much any technical work I've written, Tiago Pimentel provided critical commentary and recommendations for this blogpost. Thanks, PhD sibling :) And thank you to Sander Land for catching my previous misunderstandings about the SentencePiece implentation.



