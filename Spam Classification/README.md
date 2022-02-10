# Spam Classification 

Text Classification is the task of categorizing text documents (a document can be a tweet, an SMS, an actual document, comments etc) into predefined class labels. Here we use the SMS Spam Collection Dataset to classify an SMS as spam or ham.

## Dataset

The dataset used is the [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset) from Kaggle. The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.The files contain one message per line. Each line is composed by two columns: 

1.**Spam:** A spam SMS example

2.**Ham:** A legitimate SMS example


## Model Architecture - Naive Bayes Classifier

In statistics, Naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem of Conditional Probability with strong (naïve) independence assumptions between the features. The assumption is that feature exists completely independent of each other. They are among the simplest Bayesian network models, but coupled with kernel density estimation, they can achieve higher accuracy levels.
