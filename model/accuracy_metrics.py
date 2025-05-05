import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None) 

# Compare Accuracy Scores
def compare_accuracy_metrics(cfg, result_dict_test):
    df_result_test = pd.DataFrame.from_dict(result_dict_test,orient = "index", columns=[ 'train_accuracy', 'test_r2_score', 
                    'accuracy', 'benign_precision', 'anomaly_precision', 'recall', 'f1_score', 'FPR'])
    print(df_result_test)
    # df_result_test.to_csv('Test set metrics.csv')

    # df_result_test = pd.DataFrame.from_dict(result_dict_test,orient = "index",columns=["Score"])
    # df_result_test

    # Display the accuracy scores
    fig,ax = plt.subplots(1,2,figsize=(20,16))
    sns.barplot(x = df_result_test.index,y = df_result_test.recall,ax = ax[0])
    sns.barplot(x = df_result_test.index,y = df_result_test.FPR,ax = ax[1])
    ax[0].set_xticklabels(df_result_test.index,rotation = 85)
    ax[1].set_xticklabels(df_result_test.index,rotation = 85)

    save_path = f"{cfg.plots_dir}/Recall and FPR for Baseline.png"
    plt.savefig(save_path, dpi = 300)

    fig,ax = plt.subplots(1,2,figsize=(20,16))
    sns.barplot(x = df_result_test.index,y = df_result_test.benign_precision,ax = ax[0])
    sns.barplot(x = df_result_test.index,y = df_result_test.anomaly_precision,ax = ax[1])
    ax[0].set_xticklabels(df_result_test.index,rotation = 85)
    ax[1].set_xticklabels(df_result_test.index,rotation = 85)
    
    save_path = f"{cfg.plots_dir}/Class wise Precision for Baseline.png"
    plt.savefig(save_path, dpi=300)
    # plt.show()
