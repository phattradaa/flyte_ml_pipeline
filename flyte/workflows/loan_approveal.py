"""A basic Flyte project template that uses ImageSpec"""

import typing
from flytekit import conditional, dynamic, task, workflow ,map_task
from flytekit.experimental import eager

import pandas as pd

from sklearn.utils import resample
# import category_encoders as ce
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score

from typing import Tuple,Union,Dict,List

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# image_definition = ImageSpec(
#    name="flytekit",  # default docker image name.
#    base_image="ghcr.io/flyteorg/flytekit:py3.11-1.10.2",  # the base image that flytekit will use to build your image.
#    packages=["pandas"],  # python packages to install.
#    registry="ghcr.io/unionai-oss", # the registry your image will be pushed to.
#    python_version="3.11"  # Optional if python is installed in the base image.
# )


@task()
def data_loader(path: str) -> pd.DataFrame:
  data = pd.read_csv(path)
  return data

@task()
def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
  data['person_gender'] = data['person_gender'].map({'female':0,'male':1})
  data['previous_loan_defaults_on_file'] = data['previous_loan_defaults_on_file'].map({'No':0,'Yes':1})
  data['person_education'] = data['person_education'].map({'High School':1,
                                                         'Associate':2,
                                                         'Bachelor':3,
                                                         'Master':4,
                                                         'Doctorate':5})
  data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent'], drop_first=True)
  median_age = data['person_age'].median()
  data['person_age'] = data['person_age'].apply(lambda x: median_age if x > 100 else x)
  # data.head()
  return data

@task()
def train_test_splitter(data:pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
  X = data.drop(['loan_status'], axis=1)
  y = data['loan_status']
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
  # print(type(X_train),type(X_val),type(y_train),type(y_val))
  return X_train, X_val, y_train, y_val

@task()
def robust_scaler(X_train:pd.DataFrame,X_val:pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
  scaler = RobustScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_val_scaled = scaler.fit_transform(X_val)
  return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_val_scaled, columns=X_val.columns)

@task  
def logistics_regression_train_model(X_train:pd.DataFrame,y_train:pd.Series) -> LogisticRegression:
  model  = LogisticRegression(max_iter=1000, random_state=42)
  # print(model)
  model.fit(X_train,y_train)
  # print(model)
  return model

@task
def logistic_regression_evaluate_model(model:LogisticRegression, X_train:pd.DataFrame, X_val:pd.DataFrame, y_train:pd.Series, y_val:pd.Series) -> float:
  # print(type(model))
  # {'LighGBM' : model}
  # model = model_dict['model']
  y_val_pred = model.predict(X_val)
  # train_score = model.score(X_train,y_train)
  # test_score = model.score(X_val, y_val)
  accuracy = accuracy_score(y_val, y_val_pred)
  # results = {
  #   'Model' : "Logistic Regression",
  #   'Train Score' : train_score,
  #   'Test Score' : test_score,
  #   'Accuracy Score' : accuracy
  # }
  # ans = float(results['Accuracy Score'])
  # print(ans,type(ans))
  return accuracy_score(y_val, y_val_pred)

@task
def random_forest_train_model(X_train:pd.DataFrame,y_train:pd.Series) -> RandomForestClassifier:
  model  = RandomForestClassifier(random_state=42)
  # print(model)
  model.fit(X_train,y_train)
  # print(model)
  return model

@task
def random_forest_evaluate_model(model:RandomForestClassifier, X_train:pd.DataFrame, X_val:pd.DataFrame, y_train:pd.Series, y_val:pd.Series) -> float:
  # print(type(model))
  # {'LighGBM' : model}
  y_val_pred = model.predict(X_val)
  train_score = model.score(X_train,y_train)
  test_score = model.score(X_val, y_val)
  accuracy = accuracy_score(y_val, y_val_pred)
  results = {
    'Model' : "Random Forest",
    'Train Score' : train_score,
    'Test Score' : test_score,
    'Accuracy Score' : accuracy
  }
  ans = float(results['Accuracy Score'])
  return ans

@task
def lightgbm_train_model(X_train:pd.DataFrame,y_train:pd.Series) -> LGBMClassifier:
  model  = LGBMClassifier(verbosity=-1, random_state=42)
  # print(model)
  model.fit(X_train,y_train)
  # print(model)
  return model

@task
def lightgbm_evaluate_model(model:LGBMClassifier, X_train:pd.DataFrame, X_val:pd.DataFrame, y_train:pd.Series, y_val:pd.Series) -> float:
  # print(type(model))
  # {'LighGBM' : model}
  y_val_pred = model.predict(X_val)
  train_score = model.score(X_train,y_train)
  test_score = model.score(X_val, y_val)
  accuracy = accuracy_score(y_val, y_val_pred)
  results = {
    'Model' : "Light GBM",
    'Train Score' : train_score,
    'Test Score' : test_score,
    'Accuracy Score' : accuracy
  }
  ans = float(results['Accuracy Score'])
  return ans

@task
def xgboost_train_model(X_train:pd.DataFrame,y_train:pd.Series) -> XGBClassifier:
  model  = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
  model.fit(X_train,y_train)
  return model

@task
def xgboost_evaluate_model(model:XGBClassifier, X_train:pd.DataFrame, X_val:pd.DataFrame, y_train:pd.Series, y_val:pd.Series) -> float:
  # print(type(model))
  # {'LighGBM' : model}
  y_val_pred = model.predict(X_val)
  train_score = model.score(X_train,y_train)
  test_score = model.score(X_val, y_val)
  accuracy = accuracy_score(y_val, y_val_pred)
  results = {
    'Model' : "XG Boost",
    'Train Score' : train_score,
    'Test Score' : test_score,
    'Accuracy Score' : accuracy
  }
  ans = float(results['Accuracy Score'])
  return ans

@task
def define_best_model(results:pd.DataFrame) -> str: 
  best_model_row = results.loc[results['Accuracy Score'].idxmax()]
  best_model_name = best_model_row['Model']
  best_model_accuracy = best_model_row['Accuracy Score']

  return best_model_name

@task
def logistic_regression_wf(path:str) -> float:
  # print(path)
  data = data_loader(path)
  # print(data)
  norm_data = normalize_data(data)
  # print('3')
  X_train, X_val, y_train, y_val = train_test_splitter(norm_data)
  # print('4')
  X_train , X_val = robust_scaler(X_train , X_val)
  # print('5')
  model = logistics_regression_train_model(X_train,y_train)
  # print('6')
  acc_score = logistic_regression_evaluate_model(model,X_train,X_val,y_train,y_val)
  return acc_score
  # print('7')

@task
def logistic_regression_to_string(path:str) -> str:
  acc = logistic_regression_wf(path) 
  return f"Logistic Regression {acc}"

@task
def random_forest_wf(path:str) -> float:
  data = data_loader(path)
  norm_data = normalize_data(data)
  X_train, X_val, y_train, y_val = train_test_splitter(norm_data)
  # print(type(X_train),type(X_val),type(y_train),type(y_val))
  X_train , X_val = robust_scaler(X_train , X_val)
    # print(type(X_train),type(y_train))
  model = random_forest_train_model(X_train,y_train)
  acc_score = random_forest_evaluate_model(model,X_train,X_val,y_train,y_val)
  return acc_score

@task
def random_forest_to_string(path:str) -> str:
  acc = random_forest_wf(path) 
  return f"Random Forest {acc}"

@task
def xgboost_wf(path:str) -> float:
  data = data_loader(path)
  norm_data = normalize_data(data)
  X_train, X_val, y_train, y_val = train_test_splitter(norm_data)
  # print(type(X_train),type(X_val),type(y_train),type(y_val))
  X_train , X_val = robust_scaler(X_train , X_val)
    # print(type(X_train),type(y_train))
  model = xgboost_train_model(X_train,y_train)
  acc_score = xgboost_evaluate_model(model,X_train,X_val,y_train,y_val)
  return acc_score

@task
def xgboost_to_string(path:str) -> str:
  acc = xgboost_wf(path) 
  return f"XG Boost {acc}"

@task
def lightgbm_wf(path:str) -> float:
  data = data_loader(path)
  norm_data = normalize_data(data)
  X_train, X_val, y_train, y_val = train_test_splitter(norm_data)
  # print(type(X_train),type(X_val),type(y_train),type(y_val))
  X_train , X_val = robust_scaler(X_train , X_val)
    # print(type(X_train),type(y_train))
  model = lightgbm_train_model(X_train,y_train)
  acc_score = lightgbm_evaluate_model(model,X_train,X_val,y_train,y_val)
  return f'light_gbm {acc_score}'

@task
def lightgbm_to_string(path:str) -> str:
  acc = lightgbm_wf(path) 
  return f"Light GBM {acc}"

@task 
def compare(lr_result:float,rf_result:float,lgbm_result:float,xgb_result:float) -> Tuple[str,float]: 
  # print(lr_result)
  # print(lr_result.outputs)
  list_acc = [lr_result,rf_result,lgbm_result,xgb_result]
  result_num = max(list_acc)
  index_acc = list_acc.index(result_num)
  if index_acc == 0 :
    model_name = "Logistic regression"
  elif index_acc == 1 :
    model_name = "Random Forest"
  elif index_acc == 2 :
    model_name = "LGBM"
  elif index_acc == 3:
    model_name = "XGB"
  # result_model = list_accuracy.index(result_num)
  # print(result_num)
  return model_name , result_num

# @task
# def find_model_name(acc_score:float) -> str: 
#   return list_acc.index(acc_score)

@task
def all_wf(path:str) -> str:
  lr_result = logistic_regression_wf(path)
  # print(lr_result['string_value'])
  rf_result = random_forest_wf(path)
  # print(rf_result)
  lgbm_result = lightgbm_wf(path)
  xgb_result = xgboost_wf(path)
  list_acc = [lr_result,rf_result,lgbm_result,xgb_result]
  # model_name = find_model_name(acc_score,list_acc)
  # print(model_name)
  # return f"{lr_result}\n{rf_result}\n{lgbm_result}\n{xgb_result}"
  model_name , result_num = compare(lr_result,rf_result,lgbm_result,xgb_result)
  return f"{model_name} : {result_num}"
  # return results
  # print(xgb_result)
  # result = [lr_result,rf_result,lgbm_result,xgb_result]

  
  # print(lr_result.o0.get())
  # results = pd.DataFrame(results_data, columns=["Model", "Train Score", "Test Score", "Accuracy Score"])
  # best_model_name = compare(results)
  # print('eiei')

# @workflow
# def dynamic_wf(path:str,model_name:str) -> str:
#   if model_name == 'lr':
#     return logistic_regression_wf(path)
#   elif model_name == 'rf':
#     return random_forest_wf(path)
#   elif model_name == 'xg' : 
#     return xgboost_wf(path)
#   elif model_name == 'light' : 
#     return lightgbm_wf(path)
#   elif model_name == 'all':
#     return all_wf(path)
#   else : 
#     return "Model not support"
  
@workflow
def wf(path:str,model_name:str) -> str:
    return (
      conditional("wf")
      .if_(model_name.is_('lr'))
      .then(logistic_regression_to_string(path))
      .elif_(model_name.is_("rf"))
      .then(random_forest_to_string(path))
      .elif_(model_name.is_("xg"))
      .then(xgboost_to_string(path))
      .elif_(model_name.is_("light"))
      .then(lightgbm_to_string(path))
      .elif_(model_name.is_('all'))
      .then(all_wf(path))
      .else_()
      .fail("Model not supported")
      )

if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--path",type=str)
    parser.add_argument("--model_name",type=str)
    
    args = parser.parse_args()
    # print(args.path,args.model)
    # print(args.path,args.model)
    result = wf(path=args.path,model_name=args.model_name)
    
    print(result)
    # print(result)
