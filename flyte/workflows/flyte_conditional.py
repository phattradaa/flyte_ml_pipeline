import random

from flytekit import conditional, task, workflow
from flytekit.core.task import Echo

@task
def cal_circle_circumference(radius: float) -> float:
    return 2 * 3.14 * radius

@task 
def cal_circle_area(radius: float) -> float:
    return radius*radius* 3.14

@workflow
def shap_properties(radius:float) -> float:
    return (
        conditional("shape_properties")
        .if_((radius >= 0.1) & (radius < 1.0))
        .then(cal_circle_circumference(radius))
        .else_()
        .then(cal_circle_area)
    )
