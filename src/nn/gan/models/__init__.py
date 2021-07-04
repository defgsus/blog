import os
from pathlib import Path
from typing import Type

from .ae import *
from .ae_conv import *
from .dis import *
from .dis_conv import *
from .gen import *
from .gen_conv import *
from .gen_cppm import *


CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints"


def get_module(name: str) -> Type[nn.Module]:
    return globals()[name]


def store_module(m: torch.nn.Module, path: Optional[str] = None, suffix: Optional[str] = None):
    filename = get_module_checkpoint_filename(m, path=path, suffix=suffix, make_path=True)

    torch.save(m.state_dict(), filename)


def load_module(m: torch.nn.Module, path: Optional[str] = None, suffix: Optional[str] = None):
    filename = get_module_checkpoint_filename(m, path=path, suffix=suffix)

    state_dict = torch.load(filename)
    m.load_state_dict(state_dict)


def get_module_checkpoint_filename(
        m: torch.nn.Module, path: Optional[str] = None, suffix: Optional[str] = None,
        make_path: bool = True,
):
    filename = CHECKPOINT_PATH
    if path:
        filename = filename / path

    if make_path:
        os.makedirs(str(filename), exist_ok=True)

    if suffix:
        suffix = f"-{suffix}"
    else:
        suffix = ""

    return filename / f"{m.__class__.__name__}{suffix}.pt"
