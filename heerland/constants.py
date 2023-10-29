from projectfiles.constants import Constant
from pathlib import Path


class Constants(Constant):
    cache_dir: Path = Path(__file__).absolute().parent.parent.joinpath(".cache/")
    data_dir: Path = Path(__file__).absolute().parent.parent.joinpath("data/")


CONSTANTS = Constants()
