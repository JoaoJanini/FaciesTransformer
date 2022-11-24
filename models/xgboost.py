# Installing our favourite pycaret library
# This command will basically import all the modules from pycaret that are necessary for classification tasks
from pycaret.classification import *
# Setting up the classifier
# Pass the complete dataset as data and the featured to be predicted as target
clf=setup(data=df,target='NSP')
xgboost_classifier=create_model('xgboost')
# Whenenver we compare different models or build a model, the model uses deault
#hyperparameter values. Hence, we need to tune our model to get better performance

tuned_xgboost_classifier=tune_model(xgboost_classifier)
plot_model(tuned_xgboost_classifier,plot='confusion_matrix')
save_model(tuned_xgboost_classifier,"XGBOOST CLASSIFIER")
saved_model=load_model('XGBOOST CLASSIFIER')
