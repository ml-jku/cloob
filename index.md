---
layout: default
title:  "CLOOB: Modern Hopfield Networks with InfoLOOB Outperform CLIP"
description: Blog post
date:   2021-10-21 23:00:00 +0200
usemathjax: true
---


$$
\newcommand{\Ba}{\boldsymbol{a}}
\newcommand{\Bp}{\boldsymbol{p}}
\newcommand{\Bu}{\boldsymbol{u}}
\newcommand{\Bv}{\boldsymbol{v}}
\newcommand{\Bx}{\boldsymbol{x}}
\newcommand{\By}{\boldsymbol{y}}

\newcommand{\BU}{\boldsymbol{U}}
\newcommand{\BV}{\boldsymbol{V}}
\newcommand{\BX}{\boldsymbol{X}}
\newcommand{\BY}{\boldsymbol{Y}}

\newcommand{\soft}{\mathrm{softmax}}

\newcommand{\rL}{\mathrm{L}}
$$


This post explains the paper "[CLOOB: Modern Hopfield Networks with InfoLOOB Outperform CLIP][arxiv-paper]".


CLOOB (Contrastive Leave One Out Boost) is a novel self-supervised learning method,
where [modern Hopfield networks](#mhn) boost
contrastive learning using upper bounds on the mutual information like [InfoLOOB](#infoloob).
**CLOOB consistently outperforms CLIP** at zero-shot transfer learning
across different architectures and datasets.

A sketch of CLOOB and zero-shot transfer learning results for [CLOOB][arxiv-paper] and [CLIP][radford:21-paper] (YFCC pretraining):


{:refdef: style="text-align: center;"}
![not found](/assets/ArchTab.png){:width="800px"}
{: refdef}

# Table of Contents
1. [Success of Self-Supervised, Contrastive Learning and CLIP](#success)
2. [CLIP Suffers From Explaining Away](#explainAway)
3. [Explaining Away Hampers Applications to the Real World](#explainAwayHampers)
4. [Humans are not Affected by the Explaining Away Problem](#explainAwayHumans)
5. [Modern Hopfield Networks Solve the Explaining Away Problem](#mhn)
6. [Modern Hopfield Networks Help to Utilize InfoLOOB](#infoloob)
7. [CLOOB: Modern Hopfield Networks with InfoLOOB](#cloob)
8. [Experiments](#experiments)
9. [Code and Paper](#codeAndPaper)
10. [Additional Material](#material)
11. [Correspondence](#correspondence)


## Success of Self-Supervised, Contrastive Learning and CLIP<a name="success"></a>


Self-supervised learning based on contrastive objective functions
has become highly successful. 
An unimodal contrastive objective for pretraining of vision models contrasts views 
of images with views of other images.
A multimodal objective would for a given image contrast the corresponding caption to captions of other images or, the other way around, for a given caption contrast the corresponding image to images of other captions.


CLIP (Contrastive Language-Image Pre-training),
a recent multimodal approach, 
yielded very impressive results at
zero-shot transfer learning ([Radford et al., 2021][radford:21-paper]).
CLIP learns expressive image embeddings directly from associated language. 
These rich embeddings empower CLIP
to become 
one of the most important foundation models  ([Bommasani et al., 2021][bommasani:21-paper]).


## CLIP Suffers From Explaining Away <a name="explainAway"></a>

Though CLIP is a big success at zero-shot transfer learning, it still suffers from the "explaining away" problem ([Wellman and Henrion, 1993][wellman:93-paper]).
Explaining away is the confirmation of one cause of an observed event which prevents the method from finding alternative causes.
In object detection **explaining away focuses on one feature of an object while ignoring others** that are also indicative for the object.


Explaining away is also known as shortcut learning ([Geirhos et al., 2020][geirhos:20-paper]).
The left image below is classified as *elephant* because of the texture that resembles the skin of an elephant.
The right image is classified as *sheep* since sheeps are often observed in such landscapes ([Shane, 2018][aiweirdness:18-blog]).


{:refdef: style="text-align: center;"}
![not found](/assets/shortcut_examples.png){:width="800px"}
{: refdef}

The left image below is classified as a *boat* only because of wave and horizon line features.
Therefore, boat features are explained away ([Lapuschkin et al., 2019][lapuschkin:19-paper]).

{:refdef: style="text-align: center;"}
![not found](/assets/cleverHans4_crop.png){:width="800px"}
{: refdef}

In psychology, explaining away by spurious
correlation is know as the "Clever Hans" phenomenon ([Lapuschkin et al., 2019][lapuschkin:19-paper]).
For example, the left image below is classified as a *horse* solely because of the source tag,
which is a spurious correlation for the class *horse* in this dataset.


{:refdef: style="text-align: center;"}
![not found](/assets/cleverHans_crop.png){:width="800px"}
{: refdef}



## Explaining Away Hampers Applications to the Real World <a name="explainAwayHampers"></a>

Explaining away leads to models that take only few features into account, thereby limiting their robustness and, in turn, their application to the real world. 
This is one reason why current models perform very well on standard benchmarks,
but fail at new data, new applications, deployments in the wild, 
and stress tests ([D’Amour et al., 2020][dAmour:20-paper]; [Recht et al., 2019][recht:19-paper]; [Taori et al., 2020][taori:20-paper]; [Geirhos et al., 2020][geirhos:20-paper]).
Therefore,
we require models that better exploit co-occurrences and covariance of features.

## Humans are not Affected by the Explaining Away Problem <a name="explainAwayHumans"></a>

Humans exploit contexts and co-occurences when interacting 
with their environment, therefore are not prone to explaining away issues. For example, objects like a cake can
be recognized even if blurred when a teacup is close to it.


{:refdef: style="text-align: center;"}
![not found](/assets/teahouse_real.jpg){:width="800px"}
{: refdef}

A model might misclassify the following image as an aquarium solely.
Yet the co-occurence and placement of the pillows, the bed linen and the bedside lamps together strongly suggest a bedroom to humans.

{:refdef: style="text-align: center;"}
![not found](/assets/bedroom_aquarium.jpg){:width="800px"}
{: refdef}

In cognitive science, this concept of context exploitation is known as CSTM (Conceptual short term memory) which states that humans, who perceive a stimulus,
immediately associate it with information stored in the long term
memory.
>CSTM is represented as a combination
of new perceptual information and associations
from long term memory (LTM) out of which
structures are built.
Material that is not included in the resulting
structure is quickly forgotten ([Potter, 2012][potter:12-paper]).

Consequently, the human perception is used to retrieve patterns from an associative memory. Contexts and co-occurences are amplified in the retrieval, while peculiarities of the current perception are filtered out.

## Modern Hopfield Networks Solve the Explaining Away Problem<a name="mhn"></a>

**Modern Hopfield Networks.** In analogy to this concept of context exploitation, we use modern Hopfield networks to amplify co-occurences and the covariance structure of features that encode the input sample. Modern Hopfield networks are associative memories that have much higher storage capacity than classical Hopfield networks and can retrieve patterns with one update only ([Ramsauer et al., 2021][ramsauer:21-paper]; [Widrich et al., 2020][widrich:20-paper]).

Details about modern Hopfield networks are available in the blog
[Hopfield Networks is All You Need][ml-blog-hopfield].

**Associative Memory.** Similar to the associative memory of humans,
our approach uses associative memories to amplify
co-occurences and the covariance structure.
The associative memory of our choice are modern Hopfield networks 
because of their fast retrieval and high storage capacity 
as shown in [Hopfield networks is all you need](https://arxiv.org/abs/2008.02217).
The update mechanism of modern Hopfield networks is equivalent to
the self-attention mechanism of Transformer networks.
However, modern Hopfield networks are more general
and have a broader functionality, of which
the Transformer self-attention is just one example. 
The according [Hopfield layers](https://github.com/ml-jku/hopfield-layers)
can be built in 
Deep Learning architectures for 
associating two sets, 
encoder-decoder attention,
multiple instance learning, or 
averaging and pooling operations. 
For details, see our blog [Hopfield Networks is All You Need](https://ml-jku.github.io/hopfield-layers/).

**Amplifying Co-occurences and Covariance Structures.** Instead of using features of the input sample,
we suggest to use features that
are retrieved from a modern Hopfield network that stores reference samples.
The retrieved features possess richer co-occurrences and covariance structures as they are obtained by averaging over all stored reference samples that are similar to the input sample. 
Replacing the original features by retrieved features
reinforces co-occurences that also appear in reference samples similar to the input sample.
Additionally, *spurious co-occurences* that are peculiar to the input sample are *averaged out*.


**Demonstration of the Advantages of Modern Hopfield Networks.** In the following we demonstrate the advantages of using modern Hopfield networks.
- They amplify co-occurences and covariance structures of features that encode the input sample. 
- They remove spurious co-occurences that are peculiar to the input sample.

The presence of an object in the input is represented by the activation of certain features.
For visualization purpose only, these activated features are arranged as a sketch of this object.
For example the features that represent a tea cup are arranged
to sketch a tea cup. 

In our demonstration the task is to determine the type of location.
For example we consider the location of a tea house. This location is characterized by the *Tea House Concept* consisting of objects which often occur together, like tea pot, tea cup, cake, and fork.

Images from left to right represent:
- The input in a tea house, which indicates objects observed in this location (superimposed).
- The features that are retrieved from the modern Hopfield network by the left image.
- The *Tea House Concept*.

In the memory of the modern Hopfield networks, input representations of different locations are stored including those of tea houses.
Via the retrieval the *Tea House Concept* is amplified while features peculiar to this input are averaged out. 

{:refdef: style="text-align: center;"}
![not found](/assets/sketch_tea_hopfield.png){:width="800px"}
{: refdef}
The following image shows the same for another tea house.
{:refdef: style="text-align: center;"}
![not found](/assets/sketch_tea2_hopfield.png){:width="800px"}
{: refdef}

The similarity between the two tea houses is enhanced in the retrieved features (middle image) compared to the features representing the input (left image). The similarity originates from the extracted co-occurence and covariance structure in the retrieved features. Concluding, modern Hopfield networks enhance the co-occurence and covariance structure in the retrieved features.


## Modern Hopfield Networks Help to Utilize InfoLOOB<a name="infoloob"></a>

We consider two objectives for contrastive learning, InfoNCE and InfoLOOB:


$$
\begin{align}
 \rL_{\mathrm{InfoNCE}}   \ &= \ - \
 \frac{1}{N} \  \sum_{i=1}^N \ln \frac{f(\Bx_i,\By_i)}
 {\frac{1}{N} \ \big(f(\Bx_i,\By_i) + \sum_{j=1,j\not= i}^N  f(\Bx_j,\By_i) \big) } \ , \tag{1}  \\ 
   \rL_{\mathrm{InfoLOOB}}  \ &= \ - \
 \frac{1}{N} \  \sum_{i=1}^N \ln  \frac{f(\Bx_i,\By_i)}
 {\frac{1}{N-1} \  \sum_{j=1,j\not= i}^N  f(\Bx_j,\By_i)} \ . \tag{2} \\
\end{align}
$$


In these objectives for an anchor sample $$\By_i$$,
a positive sample $$\Bx_i$$ is contrasted with
negative samples $$\Bx_j$$, 
where the training set consists of $$N$$ pairs $$\{(\Bx_1,\By_1),(\Bx_2,\By_2),\ldots,(\Bx_N,\By_N)\}$$.
In expectation the score function $$f(\Bx_i,\By_i)$$ of a matched pair ($$\Bx_i$$, $$\By_i$$) has a high value 
and the score function $$f(\Bx_j,\By_i)$$ of an unmatched pair $$(\Bx_j,\By_i)$$ has a low value.


InfoNCE like other common objectives of contrastive learning suffer from an explaining away problem, but not InfoLOOB.

### InfoNCE is Prone to the Explaining Away Problem

**Saturation of InfoNCE.**
Learning with the InfoNCE objective increases the similarity between an anchor and a positive example while simulatenously decreasing the similarity between the anchor and the negative examples.
Training hardly increases the similarity between an anchor and a positive example, if they are already very similar. 
This saturation effect even appears, when the similarity can be easily further increased. 

**Reason for Saturation of InfoNCE.** The saturation stems from the form of the InfoNCE objective, which is $$a/(a + b)$$ where $$a$$ determines the similarity of an anchor-positive pair and $$b$$ the average similarities of all samples paired with the anchor.
For a large similarity $$a$$, the InfoNCE objective saturates and increasing $$a$$ has a small effect.

**Multiplier in the Gradient of InfoNCE.** This saturation effect leads to gradients of InfoNCE, which have a common multiplier of $$(1-p_1)$$,
where $$p_1$$ is softmax similarity between
the anchor and a positive sample.
Consquently with InfoNCE learning stalls due to vanishing gradients when $$p_1$$ is close to one which, in turn, means that the anchor and a positive are similar to each other.
Therefore explaining away can also be a problem for InfoNCE as it leads to a large $$a$$ or equivalently a $$p_1$$ close to one.

### InfoLOOB to Capture Large Mutual Information

**InfoLOOB: Introduction.**
Stalled learning can be avoided by objectives of the form $$a/b$$. 
Such an objective was introduced in [Poole et al. (2019)][poole:19-paper],
derived from a variational upper bound on
the mutual information called 
"Leave one out upper bound".
In [Cheng et al. (2020)][cheng:20-paper] this bound is called "L1Out".
For simplicity, we call this bound "InfoLOOB", where LOOB is an acronym for
"Leave One Out Bound", where the objective is the negative of the bound.
The InfoLOOB objective is similar to the InfoNCE objective except that 
the denominator does not
contain a positive sample.

**InfoLOOB for Large Mutual Information.** 
The InfoLOOB bound approximates the mutual information better than InfoNCE, 
in particular for large mutual information ([Cheng et al., 2020][cheng:20-paper]).
Modern Hopfield networks amplify the covariance structures in the retrievals, which leads 
to a large mutual information between the retrieved anchor features and the retrieved positive sample features.
Therefore, the InfoLOOB objective is perfectly suited to capture this large mutual information.

### InfoLOOB has High Variance

**High Variance of InfoLOOB.**
However, the InfoLOOB objective has high variance for small $$b$$ across batches.
This high variance hampers learning and can lead to unstable learning behavior.
We tested the variance of the (negative) InfoLOOB objective on toy tasks, with samples
drawn from Gaussian distributions following ([Belghazi et al., 2018][belghazi:18-paper]; [Poole et al., 2019][poole:19-paper]; [Cheng et al., 2020][cheng:20-paper]).

**Toy Experiment Showing High Variance is Reduced by Modern Hopfield Networks.**
In this experiment, we considered deep learning architectures with and without 
modern Hopfield networks, in which
the current learning batch is stored.
The training data has mutual information of 10 and 14.
Learning parameters are optimized for the best performance on 
a validation set. The learned model is evaluated 
on different levels of mutual information.
The figure below clearly shows that modern Hopfield networks
reduce the variance of the model for all levels of mutual information.


{:refdef: style="text-align: center;"}
![not found](/assets/steps_mi10_mi14_2.svg){:width="800px"}
{: refdef}

Additionally, modern Hopfield networks also lead to a better 
approximation of the mutual information since learning is
stabilized and parameters can be chosen from a wider range.
Concluding, modern Hopfield networks enable the InfoLOOB objective by reducing its high variance and stabilizing it. 




## CLOOB: Modern Hopfield Networks with InfoLOOB <a name="cloob"></a>

CLOOB (Contrastive Leave One Out Boost) is a novel contrastive learning method 
where modern Hopfield networks boost
learning by means of the InfoLOOB objective.
CLOOB overcomes the explaining away problem of CLIP, therefore increases robustness and 
paves the way for applications to the real world.

In the InfoLOOB objective, CLOOB substitutes the features of the input sample by features that are retrieved from a modern Hopfield network storing reference samples.
InfoLOOB has the advantage:
- It captures high mutual information.
Modern Hopfield networks have two advantages:
- They enhance the co-occurence and covariance structure in the retrieved features
- They reduce the high variance of InfoLOOB and, therefore, stabilize it. 


### CLOOB Architecture <a name="architecture"></a>


A sketch of CLOOB. In this example, we sample text-image pairs where each text describes the corresponding image.

The $$i$$-th input image is mapped by an image encoder to $$\Bx_i$$ living in an embedding space. 
Analogously, the $$i$$-th input text is mapped by a text encoder to $$\By_i$$ living in the same embedding space. 
The $$(\Bx_i, \By_i)$$ are anchor-positive pairs, where
each of the two components can serve as an anchor.

The image embedding $${\Bx}_i$$ and the text embedding $$\By_i$$ 
retrieve the embeddings $$\BU_{\Bx_i}$$ and $$\BU_{\By_i}$$, respectively,
from a modern Hopfield network that stores image embeddings $$\BU$$ (two green boxes in the left block).
The retrieved image embedding $$\BU_{\Bx_i}$$ serves as anchor in order to contrast 
the positive image embedding $$\BU_{\By_i}$$ with the negative image embeddings $$\BU_{\By_j}$$ for $$j \ne i$$.
Analog, for the second modern Hopfield network
that stores text embeddings $$\BV$$ (two green boxes in the right block).

{:refdef: style="text-align: center;"}
![not found](/assets/cloob_sketch.png){:width="800px"}
{: refdef}

In the following the CLOOB objective, 
which is the sum of InfoLOOB with $$\BU_{\Bx_i}$$ as anchor (positive: $$\BU_{\By_i}$$; negatives: $$\BU_{\By_j}$$) and InfoLOOB with $$\BV_{\By_i}$$ as anchor (positive: $$\BV_{\Bx_i}$$; negatives: $$\BV_{\Bx_j}$$),
is shown:


$$
\begin{align} \label{eq:cloob}
    \rL_{\mathrm{InfoLOOB}} &= - \frac{1}{N} \ \sum_{i=1}^N \ln \frac{\exp (\tau^{-1} \ \BU_{\Bx_i}^T \BU_{\By_i})} {\sum_{j \ne  i}^N \exp (\tau^{-1} \BU_{\Bx_i}^T \BU_{\By_j})} -
     \frac{1}{N} \sum_{i=1}^N \ln \frac{\exp (\tau^{-1} \ \BV_{\Bx_i}^T \BV_{\By_i})} {\sum_{j \ne  i}^N \exp (\tau^{-1} \ \BV_{\Bx_j}^T \BV_{\By_i})} , \tag{3}
\end{align}
$$


where the following vectors are retrieved from modern Hopfield networks


$$
\begin{align}
 \BU_{\Bx_i} \ &= \ \BU \ \soft(\beta \ \BU^T \Bx_i ) \ , \quad
 \BU_{\By_i} \ = \ \BU \ \soft(\beta \ \BU^T \By_i ) \ , \label{eq:u_retrieval} \tag{4} \\
 \BV_{\Bx_i} \ &= \ \BV \ \soft(\beta \ \BV^T \Bx_i)\ , \quad
 \BV_{\By_i} \ = \ \BV \ \soft(\beta \ \BV^T \By_i )\ . \label{eq:v_retrieval} \tag{5}
\end{align}
$$


## Experiments <a name="experiments"></a>

**Methods Compared.**
We compare CLOOB to CLIP ([Radford et al., 2021][radford:21-paper]). 
We used a reimplementation of CLIP from OpenCLIP ([Ilharco et al., 2021][ilharco:21-github]) to obtain results, where results of the OpenAI CLIP were not available. 

**Evaluation.** After pretraining, we evaluate the performance of the methods on seven downstream zero-shot transfer learning tasks with respect to their accuracy.  



### Conceptual Captions Pretraining
**Pretraining Dataset.**
The Conceptual Captions (CC) ([Sharma et al., 2018][sharma:18-paper]) dataset has 
a very rich textual description of images but only three million
image-text pairs.


**Results.** Zero-shot results for models trained on CC with ResNet-50 vision encoders for two different
checkpoints. 
Results are given as mean accuracy over 5 runs. Statistically significant results are shown in bold.
CLIP and CLOOB were trained for 31 epochs while CLIP* and CLOOB* were trained
for 128 epochs. 
**In the majority of tasks CLOOB significantly outperforms CLIP**. 

{:refdef: style="text-align: center;"}
![not found](/assets/cc_table.png){:width="700px"}
{: refdef}

### YFCC Pretraining

**Pretraining Dataset.**
The YFCC dataset, a subset of YFCC100M ([Thomee et al., 2016][thomee:16-paper]), has
15 million image-text pairs but the textual description is 
less rich than for CC and often lacks meaningful information. 


**Results Different Encoder Sizes.** Zero-shot results for the reimplementation of CLIP and CLOOB using different ResNet
architectures trained on YFCC. 
Using ResNet-50 encoders, CLOOB outperforms the CLIP
in 7 out of 8 tasks. **The performance of CLOOB scales with increased encoder size.**

{:refdef: style="text-align: center;"}
![not found](/assets/yfcc_table2.png){:width="550px"}
{: refdef}

**Results Comparing CLOOB to OpenAI CLIP.** CLOOB and CLIP trained with ResNet-50 encoder. 
**CLOOB consistently outperforms CLIP across all different downstream tasks for zero-shot**.

{:refdef: style="text-align: center;"}
![not found](/assets/yfcc_table.png){:width="550px"}
{: refdef}


## Code and Paper <a name="codeAndPaper"></a>

- [GitHub repository: CLOOB][github-repo]

- [Paper: CLOOB: Modern Hopfield Networks with InfoLOOB Outperform CLIP][arxiv-paper]

## Additional Material <a name="material"></a>

- [Paper: Hopfield Networks is All You Need][ramsauer:21-paper]

- [Blog: Hopfield Networks is All You Need][ml-blog-hopfield]

- [GitHub repository: hopfield-layers][github-hopfield]

- [Paper: Modern Hopfield Networks and Attention for Immune Repertoire Classification][widrich:20-paper]

- [Yannic Kilcher's video on modern Hopfield networks][kilcher-hopfield]

- [Blog post on Performers from a Hopfield point of view][ml-blog-performer]

- [Blog post on Energy-Based Perspective on Attention Mechanisms in Transformers][mcbal:20-blog]

For more information visit our homepage [https://ml-jku.github.io/][ml-blog].

## Correspondence <a name="correspondence"></a>

This blog post was written by Elisabeth Rumetshofer and Andreas Fürst. 

Contributions by Angela Bitto-Nemling, Michael Kopp, Johannes Lehner, Viet Tran, Günter Klambauer and Sepp Hochreiter.

Please contact us via cloob[at]ml.jku.at


[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/

[ml-blog]: https://ml-jku.github.io
[arxiv-paper]: https://arxiv.org/abs/2110.11316
[github-repo]: https://github.com/ml-jku/cloob

[ml-blog-hopfield]: https://ml-jku.github.io/hopfield-layers/
[ml-blog-performer]: https://ml-jku.github.io/blog-post-performer/
[github-hopfield]: https://github.com/ml-jku/hopfield-layers
[kilcher-hopfield]: https://www.youtube.com/watch?v=nv6oFDp6rNQ

[aiweirdness:18-blog]: https://www.aiweirdness.com/do-neural-nets-dream-of-electric-18-03-02/
[belghazi:18-paper]: https://proceedings.mlr.press/v80/belghazi18a.html
[bommasani:21-paper]: https://arxiv.org/abs/2108.07258
[cheng:20-paper]: https://proceedings.mlr.press/v119/cheng20b.html
[dAmour:20-paper]: https://arxiv.org/abs/2011.03395
[geirhos:20-paper]: https://www.nature.com/articles/s42256-020-00257-z  
[ilharco:21-github]: https://github.com/mlfoundations/open_clip
[lapuschkin:19-paper]: https://www.nature.com/articles/s41467-019-08987-4 
[mcbal:20-blog]: https://mcbal.github.io/post/an-energy-based-perspective-on-attention-mechanisms-in-transformers
[poole:19-paper]: http://proceedings.mlr.press/v97/poole19a.html
[potter:12-paper]: https://www.frontiersin.org/articles/10.3389/fpsyg.2012.00113/full
[radford:21-paper]: http://proceedings.mlr.press/v139/radford21a.html
[ramsauer:21-paper]: https://openreview.net/forum?id=tL89RnzIiCd  
[recht:19-paper]: http://proceedings.mlr.press/v97/recht19a.html
[sharma:18-paper]: https://aclanthology.org/P18-1238/
[taori:20-paper]: https://proceedings.neurips.cc/paper/2020/hash/d8330f857a17c53d217014ee776bfd50-Abstract.html
[thomee:16-paper]: https://dl.acm.org/doi/10.1145/2812802
[wellman:93-paper]: https://openreview.net/forum?id=rs-D3yrld6H
[widrich:20-paper]: https://arxiv.org/abs/2007.13505






