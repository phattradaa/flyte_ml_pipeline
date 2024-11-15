from typing import Tuple

from flytekit import conditional, dynamic, task, workflow


@task
def split(numbers: list[int]) -> Tuple[list[int], list[int], int, int]:
    return (
        numbers[0 : int(len(numbers) / 2)],
        numbers[int(len(numbers) / 2) :],
        int(len(numbers) / 2),
        int(len(numbers)) - int(len(numbers) / 2),
    )


@task
def merge(sorted_list1: list[int], sorted_list2: list[int]) -> list[int]:
    result = []
    while len(sorted_list1) > 0 and len(sorted_list2) > 0:
        # Compare the current element of the first array with the current element of the second array.
        # If the element in the first array is smaller, append it to the result and increment the first array index.
        # Otherwise, do the same with the second array.
        if sorted_list1[0] < sorted_list2[0]:
            result.append(sorted_list1.pop(0))
        else:
            result.append(sorted_list2.pop(0))

    # Extend the result with the remaining elements from both arrays
    result.extend(sorted_list1)
    result.extend(sorted_list2)

    return result


@task
def sort_locally(numbers: list[int]) -> list[int]:
    return sorted(numbers)


@dynamic
def merge_sort_remotely(numbers: list[int], run_local_at_count: int) -> list[int]:
    split1, split2, new_count1, new_count2 = split(numbers=numbers)
    sorted1 = merge_sort(numbers=split1, numbers_count=new_count1, run_local_at_count=run_local_at_count)
    sorted2 = merge_sort(numbers=split2, numbers_count=new_count2, run_local_at_count=run_local_at_count)
    return merge(sorted_list1=sorted1, sorted_list2=sorted2)


@workflow
def merge_sort(numbers: list[int], numbers_count: int, run_local_at_count: int = 5) -> list[int]:
    return (
        conditional("terminal_case")
        .if_(numbers_count <= run_local_at_count)
        .then(sort_locally(numbers=numbers))
        .else_()
        .then(merge_sort_remotely(numbers=numbers, run_local_at_count=run_local_at_count))
    )
    
if __name__ == "__main__":
    # Execute the workflow by invoking it like a function and passing in
    # the necessary parameters
    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    # parser.add_argument("--number",type=str)
    
    # args = parser.parse_args()
    numbers_list = [0,2,1,3,4,5]
    print(f"Running workflow ...")
    result = merge_sort(numbers=numbers_list,numbers_count=len(numbers_list),run_local_at_count=5)
    print(result)