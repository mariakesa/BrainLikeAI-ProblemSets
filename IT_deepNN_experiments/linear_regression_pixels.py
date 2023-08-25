from utils import load_it_data, visualize_img
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import sys
import mlflow
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
experiment = mlflow.create_experiment(experiment_name)

# Build a zscoring function

def eval_metrics(actual, pred):
    rmse_neurons = np.sqrt(mean_squared_error(
        actual, pred, multioutput='raw_values'))
    r2_neurons = r2_score(actual, pred, multioutput='raw_values')
    variance_explained_neurons = explained_variance_score(
        actual, pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    variance_explained = explained_variance_score(actual, pred)
    return {'RMSE': rmse, 'R2':r2, 'Variance explained': variance_explained, 'RMSE neurons': rmse_neurons, 
        'R2 neurons': r2_neurons, 'Variance explained neurons': variance_explained_neurons}

def plot_metric_distributions(metric_neurons):
    plt.hist(metric_neurons, bins=20, alpha=0.5, label='RMSE')
    plt.xlabel('Metric Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('Distribution of Metrics for Neurons')
    plt.show()

with mlflow.start_run(experiment_id=experiment.experiment_id):
    model = LinearRegression().fit(stimulus_train, spikes_train)
    pred=model.predict(stimulus_test)
    metrics=eval_metrics(spikes_test,pred)
    for metric in metrics.keys():
        mlflow.log_metric(metric, metrics[metric])
    
