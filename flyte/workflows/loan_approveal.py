"""A basic Flyte project template that uses ImageSpec"""

import typing
from flytekit import task, workflow

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

from typing import Tuple,List
import numpy as np
import warnings
warnings.filterwarnings("ignore")
"""
ImageSpec is a way to specify a container image configuration without a
Dockerfile. To use ImageSpec:
1. Add ImageSpec to the flytekit import line.
2. Uncomment the ImageSpec definition below and modify as needed.
3. If needed, create additional image definitions.
4. Set the container_image parameter on tasks that need a specific image, e.g.
`@task(container_image=basic_image)`. If no container_image is specified,
flytekit will use the default Docker image at
https://github.com/flyteorg/flytekit/pkgs/container/flytekit.

For more information, see the
`ImageSpec documentation <https://docs.flyte.org/projects/cookbook/en/latest/auto_examples/customizing_dependencies/image_spec.html#image-spec-example>`__.
"""

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(verbosity=-1, random_state=42)
}

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
  
@task()
def train_model(X_train:pd.DataFrame,y_train:pd.Series) -> XGBClassifier:
  model  = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
  # print(model)
  model.fit(X_train,y_train)
  # print(model)
  return model

@task()
def evaluate_model(model:XGBClassifier, X_train:pd.DataFrame, X_val:pd.DataFrame, y_train:pd.Series, y_val:pd.Series) -> dict:
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
  
  return results

@workflow
def wf(path:str) -> dict:
    data = data_loader(path)
    norm_data = normalize_data(data)
    X_train, X_val, y_train, y_val = train_test_splitter(norm_data)
    # print(type(X_train),type(X_val),type(y_train),type(y_val))
    X_train , X_val = robust_scaler(X_train , X_val)
    # print(type(X_train),type(y_train))
    model = train_model(X_train,y_train)
    result = evaluate_model(model,X_train,X_val,y_train,y_val)
    return result


if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--path",type=str)
    
    args = parser.parse_args()
    
    print(f"Running workflow ...")
    result = wf(path=args.name)
    print(result)
