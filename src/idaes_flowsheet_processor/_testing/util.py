from __future__ import (
    annotations,
)  # allows using A | B syntax for unions in Python < 3.10
import importlib
import sys
from functools import singledispatch
from importlib.metadata import entry_points
from importlib.metadata import EntryPoint
from pathlib import Path
from types import ModuleType

import pytest


def get_module_name(file_path: Path):
    file_path = file_path.resolve()
    syspath_paths = (Path(p).resolve() for p in sys.path)
    matching = []
    for syspath_path in syspath_paths:
        if file_path.is_relative_to(syspath_path):
            matching.append(syspath_path)
    if not matching:
        return None
    best_match = sorted(matching, key=lambda p: len(p.parents), reverse=True)[0]
    rel_path = file_path.relative_to(best_match)
    modname_fragments = rel_path.with_suffix("").parts
    return ".".join(modname_fragments)


@singledispatch
def get_module_path(module: ModuleType) -> Path:
    # TODO: handle packages
    return Path(module.__file__)


@get_module_path.register
def _(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    return Path(spec.origin)


def get_entry_points(group_name: str) -> list[EntryPoint]:
    all_eps = entry_points()
    try:
        # this happens for Python 3.10+
        eps = all_eps.select(group=group_name)
    except AttributeError:
        # this will happen on Python 3.9, where entry_points() has dict-like group selection
        eps = all_eps[group_name]
    # creating a temp dict with eps as keys and placeholder values
    # lets us remove duplicates while preserving insertion order (ordered set)
    # since EntryPoint instances are hashable
    unique_eps = list({ep: ... for ep in eps}.keys())
    return unique_eps


def get_test_class(cls: type | str, parent: pytest.Collector) -> pytest.Class:
    if isinstance(cls, type):
        cls_obj = cls
        mod_name = cls_obj.__module__
        cls_name = cls_obj.__name__
    else:
        mod_name, cls_name = cls.split(":")
        mod_obj = importlib.import_module(mod_name)
        cls_obj = getattr(mod_obj, cls_name)

    mod_path = get_module_path(mod_name)
    mod_collector = parent.ihook.pytest_pycollect_makemodule(
        module_path=mod_path,
        parent=parent,
    )
    cls_collector = pytest.Class.from_parent(
        mod_collector,
        name=cls_name,
        obj=cls_obj,
    )
    return cls_collector
