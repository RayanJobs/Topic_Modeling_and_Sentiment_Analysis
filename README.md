# Topic Modeling and Sentiment Analysis


Abstract: Sentiment analysis is a topic that has commanded the attention of natural language processing groups. Generally, e-commerce businesses perform sentiment analysis to assess public opinion on their products (Dang et al., 2020), while customers find reviews of such products useful when making purchasing decisions. We examine sentiment analysis algorithms on a dataset from the retailer Amazon (largest online retailer by a significant margin), recommend a most accurate model for sentiment analysis, and propose novel topic modeling techniques (based on a frequency, transformers and GPT-3 approaches) to isolate sources of satisfaction and causes of failure for various products.
Keywords: Sentiment Analysis (SA), E-commerce (EC), Naïve Bayes (NB), Random Forest (RF), FastText, Logistic Regression (LR), GPT-3, Transformers, BERT, Glove, Latent Dirichlet Allocation (LDA), Transformers, GPT-3



The dataset was obtained from: 
https://nijianmo.github.io/amazon/index.html

Focus was on Home and Kitchen in this paper. 

File xaa from a split was also provided in JSON format. 

Additional models and files attached: 
model.bin
Pre-trained fastText to assist with reproducing results. With necessary installation, results should be reproducible without loading our model. Note that FastText takes 3-5 hours to train on an 80% split. 


References
Code for this project and additional files can be found on
https://github.com/RayanJobs/Topic_Modeling_and_Sentiment_Analysis

[1]: Dataset: https://nijianmo.github.io/amazon/index.html

[2]: Amazon category JSON file: https://www.browsenodes.com/amazon.com/browseNodeLookup/1063498.html 

[3]: Glove 200-d obtained from https://nlp.stanford.edu/projects/glove/

[4]: BERTopic model obtained from https://maartengr.github.io/BERTopic/index.html [5]: FastText model obtained from https://fasttext.cc/
Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.

Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794

Hoffman et al. (2010). "Online Learning for Latent Dirichlet Allocation".

Hu, X., Wang, R., Zhou, D., & Xiong, Y. (2020). Neural topic modeling with cycle-consistent adversarial training. arXiv preprint arXiv:2009.13971.

Ramage, D., Rosen, E., Chuang, J., Manning, C. D., & McFarland, D. A. (2009, December). Topic modeling for the social sciences. In NIPS 2009 workshop on applications for topic models: text and beyond (Vol. 5, pp. 1-4).

Syed, S., & Spruit, M. (2017, October). Full-text or abstract? examining topic coherence scores using latent dirichlet allocation. In 2017 IEEE International conference on data science and advanced analytics (DSAA) (pp. 165-174). IEEE.

Székely, N., & Vom Brocke, J. (2017). What can we learn from corporate sustainability reporting? Deriving propositions for research and practice from over 9,500 corporate sustainability reports published between 1999 and 2015 using topic modelling technique. PloS one, 12(4), e0174807.
Wang, Y., Mukherjee, S., Chu, H., Tu, Y., Wu, M., Gao, J., & Awadallah, A. H. (2020). Adaptive self-training for few-shot neural sequence labeling. arXiv preprint arXiv:2010.03680.
