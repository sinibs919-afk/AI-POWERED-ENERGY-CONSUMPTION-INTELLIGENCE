import mlflow

from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score,precision_score,recall_score,f1_score,roc_auc_score)



# FORECASTING METRICS
def evaluate_forecasting_model(y_test, y_pred):

    mae = mean_absolute_error(y_test,y_pred)

    rmse = mean_squared_error(y_test,y_pred) ** 0.5

    r2 = r2_score(y_test,y_pred)

    return {"MAE": mae,"RMSE": rmse,"R2": r2}


# LOG FORECASTING METRICS
def log_forecasting_metrics(model_name,params,mae,rmse,r2):

    mlflow.log_param("model",model_name)

    mlflow.log_params(params)

    mlflow.log_metric("MAE",mae)

    mlflow.log_metric("RMSE",rmse)

    mlflow.log_metric("R2",r2)


# ANOMALY DETECTION METRICS
def evaluate_anomaly_model(y_test,y_pred,anomaly_scores):

    precision = precision_score(y_test,y_pred,zero_division=0)

    recall = recall_score(y_test,y_pred,zero_division=0)

    f1 = f1_score(y_test,y_pred,zero_division=0)

    # ROC-AUC uses continuous anomaly scores
    roc_auc = roc_auc_score(y_test,anomaly_scores)

    return {"precision": precision,"recall": recall,"f1_score": f1,"roc_auc": roc_auc}


# LOG ANOMALY METRICS TO MLFLOW
def log_anomaly_metrics(model_name,params,precision,recall,f1,roc_auc):

    mlflow.log_param("model",model_name)

    mlflow.log_params(params)

    mlflow.log_metric("precision", precision)

    mlflow.log_metric("recall",recall)

    mlflow.log_metric("f1_score",f1)

    mlflow.log_metric("roc_auc",roc_auc)