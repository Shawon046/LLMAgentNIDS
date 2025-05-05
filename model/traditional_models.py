import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, precision_recall_fscore_support, f1_score, ConfusionMatrixDisplay, confusion_matrix,precision_score, recall_score,accuracy_score 
from sklearn.metrics import roc_curve, auc,RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
from sklearn.model_selection import cross_val_score
import numpy as np 
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def baseline_traditional_ml_models(cfg, X_train, y_train, X_test, y_test):
    # To store results of models, we create a dictionary
    result_dict_test = {}

# models = {'Naïve Bayes Classifier': GaussianNB, 'Bernoulli Naïve Bayes Classifier': BernoulliNB, 'Decision Tree Classifier' : DecisionTreeClassifier, 'KNN Classifier': KNeighborsClassifier,
#         'Random Forest Classifier': RandomForestClassifier,'Logistic Regression': LogisticRegression}
#         # , 'Support Vector Classifier': SVC, 'Linear Regression': LinearRegression }
    # Mapping model names (strings) to actual class references
    model_map = {
        'GaussianNB': GaussianNB,
        'BernoulliNB': BernoulliNB,
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
        'RandomForestClassifier': RandomForestClassifier,
        'LogisticRegression': LogisticRegression,
    }


    # X_train, y_train = train_dataloader
    # X_test, y_test = test_dataloader
    warnings.filterwarnings("ignore")

    # for model in cfg.ml_models:
    #     print(f'{model} -> {cfg.ml_models[model]} ')
    models = cfg.ml_models
    print(f'Model dict is {models}')

    for model_name, model_str in models.items():
        comparison_metrics  = []
        model_function = model_map[model_str]
        print(model_name +' is running')
        try:
            model = model_function(random_state = 42)
        except:
            if(model_name=='Logistic Regression'):
                model = model_function(solver='lbfgs', max_iter=1000)
            model = model_function()
        accuracies = cross_val_score(model, X_train, y_train, cv=5)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)


        #Obtain accuracy
        train_accuracy = np.mean(accuracies)
        print("Train Score:",train_accuracy)
        comparison_metrics.append(train_accuracy)
        test_accuracy = model.score(X_test,y_test)
        comparison_metrics.append(test_accuracy)
        print("Test Score:",test_accuracy)
        # alternative can be - same result
        # print("Test Accuracy Score:",accuracy_score(y_test,y_pred)) 

        test_r2_score = r2_score(y_test, y_pred)
        print("R2 Score:",test_r2_score)
        comparison_metrics.append(test_r2_score)

        per_class_precision = precision_score( y_test, y_pred, average=None)
        print('Per-class precision score:', per_class_precision)
        comparison_metrics.append(per_class_precision[0])
        comparison_metrics.append(per_class_precision[1])
        # print('Recall: %.3f' % recall_score(y_test, y_pred))
        test_recall = recall_score(y_test, y_pred)
        print('Recall: ', test_recall)
        comparison_metrics.append(test_recall)

        test_f1_score = f1_score(y_test, y_pred, average='micro')

        print("Test Score (F1 - micro):",test_f1_score)
        # print("Test Score (F1 - macro):",f1_score(y_test, y_pred, average='macro'))
        # print("Test Score (F1 - weighted):",f1_score(y_test, y_pred, average='weighted'))
        comparison_metrics.append(test_f1_score)


        # Calculate the confusion matrix
        #
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        # determining the metrics
        TN = conf_matrix[0][0]
        # print("True Negative", TN)
        FN = conf_matrix[1][0]
        # print("False Negative", FN)
        TP = conf_matrix[1][1]
        print("True Positive", TP)
        FP = conf_matrix[0][1]
        print("False Positive", FP)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)

        print('False Positive Rate is: ', FPR)
        comparison_metrics.append(FPR)
        #
        # Print the confusion matrix using Matplotlib
        #
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        save_path = f"{cfg.plots_dir}/Confusion Matrix for {model_name}.png"
        plt.savefig(save_path, dpi=300)  # Save the figure with specified DPI
#     plt.axis('off')
        # plt.show()
        #Store results in the dictionaries
        
        result_dict_test[model_str] = comparison_metrics
        
    return result_dict_test
