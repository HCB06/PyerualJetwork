from tqdm import tqdm

GREY = "\033[90m"
GREEN = "\033[92m"
RESET = "\033[0m"

def loading_bars():
    bar_format_normal = "{bar} {l_bar} {remaining} {postfix}"
    bar_format_learner = "{bar} {remaining} {postfix}"
    return bar_format_normal, bar_format_learner

def get_loading_bar_style():
    return (f"{GREY}━{RESET}", f"{GREEN}━{RESET}")

def initialize_loading_bar(total, desc, ncols, bar_format, loading_bar_style=get_loading_bar_style(), leave=True):
    return tqdm(
        total=total,
        leave=leave,
        desc=desc,
        ascii=loading_bar_style,
        bar_format=bar_format,
        ncols=ncols,
    )