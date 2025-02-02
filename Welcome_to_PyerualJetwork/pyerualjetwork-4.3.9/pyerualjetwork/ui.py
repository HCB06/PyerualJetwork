from tqdm import tqdm

def loading_bars():
    
    GREEN = "\033[92m"
    RESET = "\033[0m"

    bar_format_normal = f"{GREEN}{{bar}}{GREEN} {RESET} {{l_bar}} {{remaining}} {{postfix}}"
    bar_format_learner = f"{GREEN}{{bar}}{GREEN} {RESET} {{remaining}} {{postfix}}"

    return bar_format_normal, bar_format_learner


def initialize_loading_bar(total, desc, ncols, bar_format, leave=True):
    return tqdm(
        total=total,
        leave=leave,
        desc=desc,
        ascii="▱▰",
        bar_format=bar_format,
        ncols=ncols,
    )