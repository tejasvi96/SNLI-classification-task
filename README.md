# SNLI classification
# Background

This project considers the problem on inference on the popular SNLI dataset where a pair of sentences are given and we have to classify them as entailment, neutral, negative.


Sentence1 | Type | Sentence2 | 
--- | --- | --- 
A man inspects the uniform of a figure in some EastAsian country| Contradiction | The man is sleeping |
An older and younger man smiling.| Neutral | Two men are smiling and laughing at the cats play-ing on the floor.|
A soccer game with multiple males playing. | Entailment  | Some men are playing a sport. |

# Experiment

We make use of two approaches-

1) Logistic Regression- Here we represent each sentence as a bag of words. Thus we have two inputs sentences as pair and the label as 0,1,2 (neutral,contradiction,entailment). We use one vs Rest classifier to produce the actual target classification making use of the Logistic Regression as the base model.

2) Deep Models- Here word embeddings are used to represent the sentence and LSTM is used to model the long term dependency between the words of the sentences and a final fully connected layer is used to produce the target over the 3 classes.


Refer logistic.py for the logistic regression based model.

Refer training.py for the deep learning based Model.

Accuracy scores of 77.1% were achieved in the experiment.
