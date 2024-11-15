from flytekit import LaunchPlan , current_context
# from .workflow import simple_wf
from loan_approveal import wf

standard_scale_launch_plan = LaunchPlan.get_or_create(
    wf,
    name="loan_approveal_wf",
    default_inputs={"path": 'data/loan_data.csv',
                    "model_name" : 'all'}
)