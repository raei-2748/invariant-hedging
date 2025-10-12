"""Global pytest fixtures and configuration hooks."""
import os
import re
import sys
from pathlib import Path
from typing import Dict

import yaml

import omegaconf._utils as oc_utils
import pytest

os.environ.setdefault("OMEGACONF_ALLOW_DUPLICATE_KEYS", "true")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _allow_duplicate_loader():
    class Loader(yaml.SafeLoader):
        pass

    Loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            r"""^(?:
         [-+]?[0-9]+(?:_[0-9]+)*\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?[0-9]+(?:_[0-9]+)*(?:[eE][-+]?[0-9]+)
        |\.[0-9]+(?:_[0-9]+)*(?:[eE][-+][0-9]+)?
        |[-+]?[0-9]+(?:_[0-9]+)*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    Loader.yaml_implicit_resolvers = {
        key: [
            (tag, regexp)
            for tag, regexp in resolvers
            if tag != "tag:yaml.org,2002:timestamp"
        ]
        for key, resolvers in Loader.yaml_implicit_resolvers.items()
    }
    Loader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:pathlib.Path",
        lambda loader, node: Path(*loader.construct_sequence(node)),
    )
    Loader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
        lambda loader, node: Path(*loader.construct_sequence(node)),
    )
    Loader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:pathlib.WindowsPath",
        lambda loader, node: Path(*loader.construct_sequence(node)),
    )
    return Loader


def _patched_loader():
    return _allow_duplicate_loader()


oc_utils.get_yaml_loader = _patched_loader


@pytest.fixture(scope="session")
def float_tolerance() -> Dict[str, float]:
    """Absolute/relative tolerances for deterministic torch comparisons."""

    return {"abs": 1e-6, "rel": 1e-6}
