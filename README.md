### Data extraction
This uses BeautifulSoup to parse xml tags.

### Features extraction
Use bag of words wth tfidf vectorizer. There are options to use stop words and lemmatizer from nltk when tokenizing.

### Models
Try Naive Bayes, Logistic Regression, SVC and XGBoost models.
Each model is fine tuned by grid search, evaluated using Stratified KFold with 3 folds. The best parameters set is then
applied on test set to calculate performance using f1 score.

Results of classification report are stored in results folder. After all, xgboost achived the best performance.

### Further improvement
- To improve single model performance, could do better parameter fine tune, or use n-gram with characters in vectorizer.
- Use Ensemble models to combine results from base models. One simple way is to calculate majority of votes of each topic, 
weighted by the accuracy on cross validation of each model.
- Use word2vec, fix number of words per document, then build a DNN model for classification.
- Use sentence2vec or paragraph2vec.
