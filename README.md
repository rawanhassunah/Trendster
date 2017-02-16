# Trendster
Galvanize Capstone Project: Demystifying trends and tracking topics across verticals.

## Motivation
![Alt text](https://github.com/rawanhassunah/Trendster/blob/master/images/categories.png)

<p align="center"> Since news is structured in a rigid categorical format, it may be difficult to follow a topic of interest in one snapshot. Generally, if one is interested in a topic, it tends to not exhibit this type of compartmentalization, and is actually more fluid in nature, sitting at the intersection of these categorical verticals. My idea was to create a set of tools and visualizations to help track the evolution of a topic of interest in the media over time. The topic that I chose to explore was gender equality. </p>

## Pipeline
### Filtering
![Alt text](https://github.com/rawanhassunah/Trendster/blob/master/images/pipeline1.png)

<p align="center"> I had to think extensively about how I would collect articles about my topic. The easiest solution would have been to perform a keyword search and pull out articles that mention any of the keywords. However, not only is it difficult to build a concept/theme around keywords, but depending on the keywords I chose, my subset was at risk of being biased. Given my time limit, a I chose to train a classifier on articles about my topic in order to filter out similar articles from my large article corpus. </p>
<p align="center"> • </p>
<p align="center"> I hand labeled 900 articles (100 articles about gender equality), I split my data into stratified train and test sets, and used the term frequency matrix of my train set to train a gradient boosting classifier. My model was comprised of 100 weak learners. It achieved a mean recall score of 0.78 and a mean precision score of 0.83 through cross validation. It achieved a recall and precision score of 0.85 when tested on test set.</p>
![Alt text](https://github.com/rawanhassunah/Trendster/blob/master/images/pipeline2.png)

### Topic Modeling
![Alt text](https://github.com/rawanhassunah/Trendster/blob/master/images/pipeline3.png)

## Key Takeaways
### Relevant and time dependent categories.
![Alt text](https://github.com/rawanhassunah/Trendster/blob/master/images/headlines.png)

### Meaningful nuances between topics.
![Alt text](https://github.com/rawanhassunah/Trendster/blob/master/images/lawsuits.png)
![Alt text](https://github.com/rawanhassunah/Trendster/blob/master/images/sh.png)
