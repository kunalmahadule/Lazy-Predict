# 3 April 2025
# for more: https://pypi.org/project/lazypredict/ 


import lazypredict
'''
In Machine Learning every time we need to test result with different different parameters
for different different models but with the help of lazypredict we can overcome this 
headache it gives directly the performace result for all ML algo.

'''



from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer # sklearn also has it's own dataset
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)


'''

Model                             Accuracy  ...  Time Taken
                                    ...            
LinearSVC                          0.99  ...        0.01
Perceptron                         0.99  ...        0.01
LogisticRegression                 0.99  ...        0.01
XGBClassifier                      0.98  ...        0.07
SVC                                0.98  ...        0.01
LabelPropagation                   0.98  ...        0.02
LabelSpreading                     0.98  ...        0.01
BaggingClassifier                  0.97  ...        0.05
PassiveAggressiveClassifier        0.98  ...        0.01
SGDClassifier                      0.98  ...        0.01
RandomForestClassifier             0.97  ...        0.15
CalibratedClassifierCV             0.98  ...        0.04
LGBMClassifier                     0.97  ...        0.07
QuadraticDiscriminantAnalysis      0.96  ...        0.02
ExtraTreesClassifier               0.97  ...        0.08
RidgeClassifierCV                  0.97  ...        0.01
RidgeClassifier                    0.97  ...        0.02
AdaBoostClassifier                 0.96  ...        0.11
KNeighborsClassifier               0.96  ...        0.13
BernoulliNB                        0.95  ...        0.02
LinearDiscriminantAnalysis         0.96  ...        0.05
GaussianNB                         0.95  ...        0.00
NuSVC                              0.95  ...        0.02
ExtraTreeClassifier                0.94  ...        0.01
NearestCentroid                    0.95  ...        0.01
DecisionTreeClassifier             0.93  ...        0.01
DummyClassifier                    0.64  ...        0.01

'''