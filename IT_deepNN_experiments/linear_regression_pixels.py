from utils import load_it_data, visualize_img
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import sys
import mlflow
import time
# sys.path.append('./../')

# Insert the folder where the data is, if you download in the same folder as this notebook then leave it blank
path_to_data = '/home/maria/BrainLikeAI-ProblemSets/IT_deepNN_experiments/'

stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = load_it_data(
    path_to_data)

# Develop a linear regression model that predict the neural activity from pixels.
# You can try out different types of linear regression (ridge, least-square regression)
# Evaluate your prediction (Check both the correlation and explained variance for each neuron).
# Plot the distribution for the explained variance across neurons.
stimulus_train = np.mean(stimulus_train, axis=1).reshape(-1, 224*224)
stimulus_test = np.mean(stimulus_test, axis=1).reshape(-1, 224*224)
stimulus_val = np.mean(stimulus_val, axis=1).reshape(-1, 224*224)

experiment_name = "linear_regression_pixels"
#experiment = mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Build a zscoring function

def eval_metrics(pred_dct):
    metrics={'train':{},'val':{}}
    for k in pred_dct.keys():
        rmse_neurons = np.sqrt(mean_squared_error(
            pred_dct[k][0], pred_dct[k][1], multioutput='raw_values'))
        metrics[k]['rmse_neurons']=rmse_neurons
        r2_neurons = r2_score(pred_dct[k][0], pred_dct[k][1], multioutput='raw_values')
        metrics[k]['r2_neurons']=r2_neurons
        variance_explained_neurons = explained_variance_score(
            pred_dct[k][0], pred_dct[k][1], multioutput='raw_values')
        metrics[k]['variance_explained_neurons']=variance_explained_neurons
        rmse = np.sqrt(mean_squared_error(pred_dct[k][0], pred_dct[k][1]))
        metrics[k]['rmse']=rmse
        r2 = r2_score(pred_dct[k][0], pred_dct[k][1])
        metrics[k]['r2']=r2
        variance_explained = explained_variance_score(pred_dct[k][0], pred_dct[k][1])
        metrics[k]['variance_explained']=variance_explained
    return metrics

def plot_metric_distributions(metrics, model_str):
    #plt.switch_backend('agg')  
    train_dct = metrics['train']
    val_dct = metrics['val']
    figure_paths = []

    for k in train_dct:
        if k in ['rmse_neurons', 'r2_neurons', 'variance_explained_neurons']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            ax1.hist(train_dct[k], bins=20, alpha=0.5, color='blue', label='Train')
            ax1.set_title('Distribution of metric ' + k + ' (Train)')
            ax1.legend()

            ax2.hist(val_dct[k], bins=20, alpha=0.5, color='pink', label='Validation')
            ax2.set_title('Distribution of metric ' + k + ' (Validation)')
            ax2.legend()

            plt.tight_layout()
            
            '''
            plt.figure()  # Create a new plot for each metric
            #print(k, train_dct[k], val_dct[k])
            plt.hist(train_dct[k], bins=2, alpha=0.5, label='train', color='pink')
            #plt.hist(val_dct[k], bins=20, alpha=0.5, label='validation')
            plt.xlabel('Metric Value')
            plt.ylabel('Frequency')
            plt.legend(loc='upper right')
            plt.title('Distribution of metric ' + k)
            plt.show()
            '''
                       
            suffix = f"{int(time.time())}"
            histogram_filepath = '/media/maria/DATA/MLFlow_plots/' + model_str + '_' + k + '_' + suffix + '.png'
            plt.savefig(histogram_filepath)
            plt.close()

            figure_paths.append(histogram_filepath)

    return figure_paths

    

with mlflow.start_run():
    model = LinearRegression().fit(stimulus_train, spikes_train)
    pred_train=model.predict(stimulus_train)
    pred_val=model.predict(stimulus_val)
    #print(pred_train, pred_val)
    pred_dct={'train': [spikes_train, pred_train], 'val': [spikes_val, pred_val]}
    metrics=eval_metrics(pred_dct)
    #print(metrics)
    for split in metrics.keys():
        for m in metrics[split]:
            if m=='rmse' or m=='r2' or m=='variance_explained': 
                mlflow.log_metric(split+'_'+m, metrics[split][m])
    figure_paths=plot_metric_distributions(metrics, model_str='LR_pixel')
    for path in figure_paths:
        mlflow.log_artifact(path)
    
