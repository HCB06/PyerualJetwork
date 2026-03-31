from tqdm import tqdm

GREY  = "\033[90m"
PURPLE = "\033[35m"
RESET = "\033[0m"


def loading_bars():
    bar_format_normal  = "{bar} {l_bar} {remaining} {postfix}"
    bar_format_learner = "{bar} {remaining} {postfix}"
    # YENİ: 4 bilgi (Epoch, Train Accuracy, Train Loss, Batch) tek satırda
    bar_format_learner_batch = "{bar} {remaining} {postfix}"
    return bar_format_normal, bar_format_learner, bar_format_learner_batch


def get_loading_bar_style():
    return (f"{GREY}━{RESET}", f"{PURPLE}━{RESET}")


def initialize_loading_bar(total, desc, ncols, bar_format,
                           loading_bar_style=get_loading_bar_style(), leave=True, dynamic_ncols=False):
    """Mevcut bar — değişmedi."""
    return tqdm(
        total=total,
        leave=leave,
        desc=desc,
        ascii=loading_bar_style,
        bar_format=bar_format,
        ncols=ncols,
        dynamic_ncols=dynamic_ncols
    )

