from .dis import *
from .dis_conv import *
from .gen import *
from .gen_conv import *
from .gen_cppm import *


def get_module(name: str) -> nn.Module:
    return globals()[name]
