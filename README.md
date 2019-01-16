# Unsupervised-morphology-induction-word2vec
Implementation of the following paper for CMPT882: Neuro Machine Translation course at Simon Fraser University

Soricut, Radu, and Franz Och. "Unsupervised morphology induction using word embeddings." Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2015.
http://www.aclweb.org/anthology/N15-1186


## Abstract

Soricut and Och [2015] proposed an unsupervised, language agnostic method to extract morphological rules and build a morphological analyzer. Their model improved on the state of art for word similarity in the morphologically-rich Stanford Rare-word dataset. For this project, I implemented the method proposed by Soricut and Och [2015] and studied the performance of their model under different word embeddings. To speed up computation, I used only top 100,000 words to extract the morphological rules. The pretrained word embeddings from Mikolov et al. [2013a,b] and Pennington et al. [2014] gave Spearman correlation of 22.9 and 22.5 respectively on RW dataset. By inducing morphological transformation and generating vector representation for rare and unknown words, I was able to improve the spearman correlation to 24.5 for SGNG and 25.9 for GloVe.
