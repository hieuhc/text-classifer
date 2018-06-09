## Description
A multi-label text classifier on Reuters 21578 dataset. Demonstrate two approaches: using vectorizer
bag of word along with tfidf, and a CNN with embedding vectors.


## Data extraction
This uses BeautifulSoup to parse xml tags.

## Vectorizer
### Features extraction
Use bag of words wth tfidf vectorizer. There are options to use stop words and lemmatizer from nltk when tokenizing.

### Models
Try Naive Bayes, Logistic Regression, SVC and XGBoost models.
Each model is fine tuned by grid search, evaluated using Stratified KFold with 3 folds. The best parameters set is then
applied on test set to calculate performance using f1 score.

Results of classification report are stored in results folder. After all, xgboost achived the best performance.

### Usage
Install package in `requirements.txt`, then
```
cd vectorizer
python model.py --help
```

### Further improvement
- To improve single model performance, could do better parameter fine tune, or use n-gram with characters in vectorizer.
- Use Ensemble models to combine results from base models. One simple way is to calculate majority of votes of each topic, 
weighted by the accuracy on cross validation of each model.


## CNN
### Model
Adopt network structure from [this article](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/),
with following changes to make it work for multi-label text classification:
- Change the loss function to `tf.nn.sigmoid_cross_entropy_with_logits`.
- Apply a function to create predicted classes based on scores.
- Evaluation on dev set at every number of steps with F score.
- Use learning rate decay to prevent local optima.

### Usage
Install package in `requirements.txt`. Modify parameters in `nn/config.yaml then
```
cd nn
python train.py
```

### Further improvement
- Currently `embeddings` are randomly initialized. We can use word2vec model to replace this.
- [Universal sentence encoder](https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/1) encodes the 
whole sentence before fitting into a CNN
