import pytest

from oiwqi_ml.config import load_config
from oiwqi_ml.synthetic import write


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory, cfg):
    return write(tmp_path_factory.mktemp("synthetic"), cfg)
