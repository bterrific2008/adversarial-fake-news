# Fake News Detection

## Overview  
The proliferation of disinformation across social media has led the application of deep learning techniques to detect fake news. However, it is difficult to understand how deep learning models make decisions on what is fake or real news, and furthermore these models are vulnerable to adversarial attacks. In this project, we test the resilience of a fake news detector against a set of adversarial attacks. Our results indicate that a deep learning model remains vulnerable to adversarial attacks, but also is alarmingly vulnerable to the use of generic attacks: the inclusion of certain sequences of text whose inclusion into nearly any text sample can cause it to be misclassified. We explore how this set of generic attacks against text classifiers can be detected, and explore how future models can be made more resilient against these attacks.

## Dataset Description

Our fake news model and dataset are taken from this [github repo](https://github.com/rockash/Fake-news-Detection).

* train.csv: A full training dataset with the following attributes:
  * id: unique id for a news article
  * title: the title of a news article
  * author: author of the news article
  * text: the text of the article; could be incomplete
  * label: a label that marks the article as potentially unreliable
    * 1: unreliable
    * 0: reliable

* test.csv: A testing training dataset with all the same attributes at train.csv without the label.

## Adversarial Text Generation
It's difficult to generate adversarial samples when working with text, which is discrete. A workaround, proposed by J. Gao et al. has been to create small text perturbations, like misspelled words, to create a black-box attack on text classification models. Another method taken by N. Papernot has been to find the gradient based off of the word embeddings of sample text. Our approach uses the algorithm proposed by Papernot to generate our adversarial samples. While Gao’s method is extremely effective, with little to no modification of the meaning of the text samples, we decided to see if we could create valid adversarial samples by changing the content of the words, instead of their text.

## Methodology
Our original goal was to create a model that could mutate text samples so that they would be misclassified by the model. We accomplished this by implementing the algorithm set out by Papernot in Crafting Adversarial Input Sequences. The proposed algorithm generates a white-box adversarial example based on the model’s Jacobian matrix. Random words from the original text sample are mutated. These mutations are determined by finding a word in the embedding where the sign of the difference between the original word and the new word are closest to the sign of the Jacobian of the original word. The resulting words have an embedding direction that most closely resemble the direction indicated as being most impactful according to the model’s Jacobian.

A fake news text sample modified to be classified as reliable is shown below:

> Council of Elders Intended to Set Up Anti-ISIS Coalition by Jason Ditz, ~~October~~ _said_ 31, 2016 Share This
ISIS has killed a number of Afghan tribal elders and wounded several more in Nangarhar Province’s main city of Jalalabad today, with a suicide bomber from the group targeting a meeting of the council of elders in the city.
The details are still scant, but ISIS claims that the council was established in part to discuss the formation of a tribal anti-ISIS coalition in the area. They claimed 15 killed and 25 wounded, labeling the victims “apostates.”
~~Afghan~~ _000_ government officials put the toll a lot lower, saying only four were killed and ~~seven~~ _mr_ wounded in the attack. Nangarhar is the main base of operations for ISIS forces in Afghanistan, though they’ve recently begun to pop up around several other provinces.
Whether the council was at the point of establishing an anti-ISIS coalition or not, this is in keeping with the ~~group~~ _mr_'s reaction to any sign of growing local resistance, with ISIS having similarly made an example of tribal groups in Iraq and Syria during their establishment there. Last 5 posts by Jason Ditz

We also discovered a phenomena where adding certain sequences of text to samples would cause them to be misclassified without needing to make any additional modifications to the original text. To discover additional sequences, we took three different approaches: generating sequences based on the sentiments of the word bank, using Papernot’s algorithm to append new sequences, and creating sequences by hand.

### Modified Papernot

Papernot’s original algorithm had been trained to mutate existing words in an input text to generate the adversarial text. However, our LSTM model pads the input, leaving spaces for blank words when the input length is small enough. We modify Papernot’s algorithm to mutate on two “blank” words at the end of our input sequence. This will generate new sequences of text that can then be applied to other samples, to see if they can serve as generic attacks.

The modified Papernot algorithm generated two-word sequences of the words ‘000’, ‘said’, and ‘mr’ in various orders, closely resembling the word substitutions created by the baseline Papernot algorithm. It can be expected that the modified Papernot will still use words identified by the baseline method, given that both models rely on the model’s Jacobian matrix when selecting replacement words. When tested against all unreliable samples, sequences generated are able to shift the model’s confidence to inaccurately classify a majority of samples as reliable instead.

### Handcraft

Our simplest approach to the generation was to manually look for sequences of text by hand. This involved looking at how the model had performed on the training set, how confident it was on certain samples, and looking for patterns in samples that had been misclassified. We tried to rely on patterns that appear to a human observer to be innocuous, but also explored other patterns that would change the meaning of the text in significant ways.

|Methodology | Sample Sequence | False Discovery Rate |
| :-: | :-: | :-: |
| Papernot | mr 000 | 0.37% |
| Papernot | said mr | 29.74% |
| Handcraft | follow twitter | 26.87% |
| Handcraft | nytimes com | 1.70% |

## Conclusion

One major issue with the deployment of deep learning models is that "[the ease with which we can switch between any two decisions in targeted attacks is still far from being understood](https://dblp.org/rec/journals/corr/abs-1901-10861.html)." It is primarily on this basis that we are skeptical of machine learning methods. We believe that there should be greater emphasis placed on identifying the set of misclassified text samples when evaluating the performance of fake news detectors. If seemingly minute perturbations in the text can change the entire classification of the sample, it is likely that these weaknesses will be found by fake news distributors, where the cost of producing fake news is cheaper than the cost of detecting it.

Our project also led to the discovery of the existence of a set of sequences that could be applied to nearly any text sample to then be misclassified by the model, resembling generic attacks from the cryptography field. We proposed a modification of Papernot’s Jacobian-based adversarial attack to automatically identify these sequences.  However, some of these generated sequences do not feel natural to the human eye, and future work can be placed into improving their generation. For now, while the eyes of a machine may be tricked by our samples, the eyes of a human can still spot the differences.

## References
  * [Original Fake News Identification - Stanford CS229](http://cs229.stanford.edu/proj2017/final-reports/5244348.pdf)
  * [Datasets from Kaggle](https://www.kaggle.com/c/fake-news/data)
  * [Deep Word Bug](https://github.com/QData/deepWordBug)
  * [Crafting Adversarial Input Sequences for Recurrent Neural Networks](https://arxiv.org/abs/1604.08275)
