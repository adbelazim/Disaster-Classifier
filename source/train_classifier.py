import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import process_tweet
import data

#this function train and generate report of the classifier
def classifier(X_train,y_train,X_test,y_test):
   vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True,use_idf = True,ngram_range=(1, 2))
   svm_clf =svm.LinearSVC(C=0.1)
   vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
   vec_clf.fit(X_train,y_train)
   joblib.dump(vec_clf, 'svmClassifier.pkl', compress=3)
   y_pred = vec_clf.predict(X_test)
   with open("report.txt",'w') as f:
      f.write(sklearn.metrics.classification_report(y_test, y_pred))


def main():
   
   #tweets,labels=data.read_data("../data/dataset_terremoto_iquique_2014.csv")
   tweets,labels=data.read_data("../data/tweets-iquique-2014-tipo-informacion.csv")
   processed_tweets = process_tweet.process_tweets(tweets)

   X_train, X_test, y_train, y_test = train_test_split(tweets,labels,test_size=0.30, random_state=42)

   classifier(X_train,y_train,X_test, y_test)

if __name__=="__main__":
   main()
