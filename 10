import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB,CategoricalNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
pipeline = make_pipeline(
    TfidfVectorizer(),
    StackingClassifier(
        estimators=[('nb', MultinomialNB()), ('svc', SVC(probability=True)), ('rf', RandomForestClassifier(n_estimators=100, random_state=42))],
        final_estimator=LogisticRegression()
    )
)
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
