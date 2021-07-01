from typing import Type

from .dis import *
from .dis_conv import *
from .gen import *
from .gen_conv import *
from .gen_cppm import *


def get_module(name: str) -> Type[nn.Module]:
    return globals()[name]
