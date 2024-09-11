"""
utils package
"""
from typing import Dict, Tuple


# def parse_microbatch_key(key: str) -> Tuple[bool, int, int]:
#     "parse microbatch key"
#     is_forward = key.startswith("f")
#     mid, pid = key.split("_")[1:]

#     return is_forward, int(pid), int(mid)

def parse_microbatch_key(key: str) -> Tuple[bool, int, int]:
    "parse microbatch key"
    flagnum = 0
    if key.startswith("f"):
        flagnum = 1
    elif key.startswith("b"):
        flagnum = 2
    elif key.startswith("w"):
        flagnum = 3
    mid, pid = key.split("_")[1:]

    return flagnum, int(pid), int(mid)

def _replace_mid_in_key(key: str, new_mid: int) -> str:
    flagnum, pid, _ = parse_microbatch_key(key)
    beginstring = ''
    if flagnum == 1:
        beginstring = 'f'
    elif flagnum == 2:
        beginstring = 'b'
    elif flagnum == 3:
        beginstring = 'w' 

    return f"{beginstring}_{new_mid}_{pid}"


def resort_microbatch_index(num_microbatches: int, model_res: Dict[str, int]) -> dict:
    "resort microbatch index"
    #print(model_res)
    max_value = max(model_res.values())
    only_forward_starts = {mb: max_value for mb in range(num_microbatches)}

    for key, value in model_res.items():
        if key.startswith("b_") or key.startswith("w_"):
            continue

        _, _, mid = parse_microbatch_key(key)

        if only_forward_starts[mid] > value:
            only_forward_starts[mid] = value #取每个mirobatch的最小值，即为每个mirobatch的开始

    #print(only_forward_starts)

    sorted_forward_starts = sorted(only_forward_starts.items(), key=lambda x: x[1])
    sorted_indexes = {
        pair[0]: new_idx for new_idx, pair in enumerate(sorted_forward_starts)
    }

    #print(sorted_indexes)

    res = {
        _replace_mid_in_key(key, sorted_indexes[parse_microbatch_key(key)[2]]): value
        for key, value in model_res.items()
    }

    #print(res)
    # print(model_res)

    return res
