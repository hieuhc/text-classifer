from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, naive_bayes, linear_model, metrics, model_selection
from xgboost.sklearn import XGBClassifier
import os


class LemmaTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        lemmatizer = WordNetLemmatizer()
        analyzer = super(TfidfVectorizer,self).build_analyzer()
        return lambda doc: (lemmatizer.lemmatize(w) for w in analyzer(doc))


class Model:
    def __init__(self, stop_words, lemmatizer):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words) if not lemmatizer else LemmaTfidfVectorizer(stop_words=stop_words)
        self.model_name = None
        self.pipeline = None
        self.parameters = None

    def train_validate_predict(self, x_train, y_train, x_test, y_test, topics_selected):
        """
        Use grid search to fine tune parameters. Make predictions. Store results to file
        :param x_train: list of train documents
        :param y_train: multi-labels train target
        :param x_test: list of test documents
        :param y_test: multi-labels test target
        :param topics_selected: Topic identification
        :return:
        """
        grid_search = model_selection.GridSearchCV(self.pipeline, self.parameters, scoring='f1_micro', cv=None, n_jobs=4,
                                                   verbose=10)
        grid_search.fit(x_train, y_train)

        print("Best parameters set:")
        print(grid_search.best_estimator_.steps)

        print("Perform best classifier on test data:")
        best_clf = grid_search.best_estimator_
        predictions = best_clf.predict(x_test)
        report = metrics.classification_report(y_test, predictions, target_names=topics_selected)
        print(report)
        with(open(os.path.join(os.getcwd(), 'results', '{}.txt'.format(self.model_name)), 'w')) as f:
            f.write('Best params: ' + str(best_clf.get_params))
            f.write('\n--------------\n')
            f.write(report)


class LRModel(Model):
    def __init__(self, model_name, stop_words, lemmatizer):
        super(LRModel, self).__init__(stop_words, lemmatizer)
        self.model_name = model_name
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('cls', OneVsRestClassifier(linear_model.LogisticRegression()))
        ])
        self.parameters = {
            'vectorizer__max_df': (0.5, 0.75),
            'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
            "cls__estimator__C": [0.01, 0.1, 1],
        }


class NBModel(Model):
    def __init__(self, model_name, stop_words, lemmatizer):
        super(NBModel, self).__init__(stop_words, lemmatizer)
        self.model_name = model_name
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('cls', OneVsRestClassifier(naive_bayes.MultinomialNB(
                fit_prior=True, class_prior=None))),
        ])
        self.parameters = {
            'vectorizer__max_df': (0.5, 0.75),
            'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'cls__estimator__alpha': (1e-2, 1e-3)
        }


class SVCModel(Model):
    def __init__(self, model_name, stop_words, lemmatizer):
        super(SVCModel, self).__init__(stop_words, lemmatizer)
        self.model_name = model_name
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('cls', OneVsRestClassifier(svm.LinearSVC())),
        ])
        self.parameters = {
            'vectorizer__max_df': (0.5, 0.75),
            'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'cls__estimator__C': [0.01, 0.1, 1]
        }


class XGBModel(Model):
    def __init__(self, model_name, stop_words, lemmatizer, max_features):
        super(XGBModel, self).__init__(stop_words, lemmatizer)
        self.model_name = model_name
        self.vectorizer.max_features = max_features
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('cls', OneVsRestClassifier(XGBClassifier())),
        ])
        self.parameters = {
            'vectorizer__max_df': (0.5, 0.75),
            'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'cls__estimator__max_depth': [3, 11],
            'cls__estimator__subsample': [0.25, 0.5, 0.75],
            'cls__estimator__colsample_bytree': [0.25, 0.5, 0.75]
        }