from __future__ import (
    annotations,
)  # allows using A | B syntax for unions in Python < 3.10
import importlib
from collections.abc import Iterable
from collections.abc import Mapping
from importlib.metadata import entry_points
from importlib.metadata import EntryPoint
from pathlib import Path
from types import ModuleType

try:
    from typing import TypeAlias
except ImportError:
    TypeAlias = type

import pytest

from idaes_flowsheet_processor.api import FlowsheetInterface
from . import util


ModuleName: TypeAlias = str


class IsValidEntryPoint(pytest.Item):

    def __init__(self, *, entry_point: EntryPoint, **kwargs):
        super().__init__(**kwargs)
        self.entry_point = entry_point

    def runtest(self):
        module = self.entry_point.load()
        assert isinstance(module, ModuleType)


class IsValidModule(pytest.Item):
    def __init__(self, *, module_name: ModuleName, **kwargs):
        super().__init__(**kwargs)
        self.module_name = module_name

    def runtest(self):
        module = importlib.import_module(self.module_name)
        assert isinstance(module, ModuleType)


class FlowsheetInterfacesCollector(pytest.Collector):
    """
    This top-level (immediately below Session) collector is mostly a proxy for the plugin instance
    to avoid having the plugin instance itself implement the pytest.Collector interface
    and be used directly in the pytest collection process.
    """

    def __init__(
        self,
        *,
        test_class: type | str,
        entry_points: list[tuple[pytest.Mark, EntryPoint]],
        modules: list[tuple[pytest.Mark, ModuleName]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._test_class = test_class
        self._entry_points = list(entry_points)
        self._modules = list(modules)

    def collect(self):
        cls_collector = util.get_test_class(
            self._test_class,
            parent=self,
        )
        for marker, ep in self._entry_points:
            item = IsValidEntryPoint.from_parent(
                self,
                name=f"{ep.group}:{ep.name}",
                entry_point=ep,
            )
            item.add_marker(marker)
            yield item
        for marker, mod in self._modules:
            item = IsValidModule.from_parent(
                self,
                name=mod,
                module_name=mod,
            )
            item.add_marker(marker)
            yield item
        yield cls_collector


class IdaesFlowsheetsPlugin:
    def __init__(self):
        self._enabled = False
        self._registered_as: str | None = None
        self._entry_points: list[tuple[pytest.Mark, EntryPoint]] = []
        self._modules: list[tuple[pytest.Mark, ModuleName]] = []
        self._marker_name: str = "flowsheet"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def pytest_addoption(
        self, parser: pytest.Parser, pluginmanager: pytest.PytestPluginManager
    ) -> None:
        name = self._registered_as = pluginmanager.get_name(self)
        group = parser.getgroup(
            name,
            description="Options for {name!r} plugin to test IDAES flowsheet interfaces",
        )
        group.addoption(
            "--idaes-flowsheets",
            help="Enable testing of IDAES flowsheets",
            default=False,
            action="store_true",
            dest="fs_enabled",
        )
        group.addoption(
            "--entry-points-group",
            help="Entry points group from where flowsheets should be loaded, e.g. 'watertap.flowsheets'",
            dest="fs_entry_points_group",
            default=None,
        )
        group.addoption(
            "--modules",
            help="One or more Python modules where flowsheet interface is defined as an absolute dotted import, e.g. 'watertap.flowsheets.my_mod_ui'",
            dest="fs_modules",
            action="extend",
            nargs="+",
        )
        group.addoption(
            "--test-class",
            help="pytest test class to use for testing each discovered flowsheet interface, e.g. 'my_package.my_flowsheet_interfaces_tests:TestMyInterface",
            dest="fs_test_class",
            default="idaes_flowsheet_processor._testing.default_tests:TestFlowsheetInterface",
        )

    def pytest_configure(self, config: pytest.Config) -> None:
        opts = config.option
        self._enabled = bool(opts.fs_enabled)
        if not self.enabled:
            config.pluginmanager.unregister(self)
            return
        if (epg := opts.fs_entry_points_group) is not None:
            self._entry_points = [
                (self.marker_for(ep), ep)
                for ep in util.get_entry_points(group_name=epg)
            ]
        if (modules := opts.fs_modules) is not None:
            self._modules = [(self.marker_for(mod), mod) for mod in modules]
        self._test_class = opts.fs_test_class
        config.addinivalue_line(
            "markers",
            f"{self._marker_name}([source,] name): applied to tests pertaining to the same flowsheet",
        )

    def marker_for(self, obj: ModuleName | EntryPoint) -> pytest.Mark:
        m = getattr(pytest.mark, self._marker_name)
        try:
            return m(obj.group, obj.value)
        except AttributeError:
            return m(obj)

    # tryfirst=True to make it show up last (after output from other plugins)
    @pytest.hookimpl(tryfirst=True)
    def pytest_report_header(self, config, start_path) -> list[str]:
        prefix = f"{self._registered_as}: "
        out = []
        own_options = {
            name: val
            for name, val in config.option.__dict__.items()
            if name.startswith("fs_")
        }
        for name, val in own_options.items():
            out.append(f"{name}: {val}")
        out.append(
            f"found {len(self._entry_points)} entry point(s)",
        )
        return [prefix + line for line in out]

    @pytest.hookimpl(wrapper=True)
    def pytest_make_collect_report(self, collector: pytest.Collector):
        # this "injects" our custom collector so that it appears as a direct child of the
        # pytest.Session instance, and doesn't interfere with other top-level collectors,
        # including the `pytest.Directory` which is the root of usual Python tests
        report: pytest.CollectReport = yield
        if isinstance(collector, pytest.Session):
            our_collector = FlowsheetInterfacesCollector.from_parent(
                collector,
                name=self._registered_as,
                test_class=self._test_class,
                entry_points=self._entry_points,
                modules=self._modules,
                nodeid=self._registered_as,
            )
            report.result.append(our_collector)
        return report

    def pytest_generate_tests(self, metafunc: pytest.Metafunc) -> None:
        if "flowsheet_interface" in metafunc.fixturenames:
            paramsets = []
            for marker, ep in self._entry_points:
                paramsets.append(
                    pytest.param(
                        ep.value,
                        id=ep.name,
                        marks=[marker],
                    )
                )
            for marker, mod in self._modules:
                paramsets.append(
                    pytest.param(
                        mod,
                        id=mod,
                        marks=[marker],
                    )
                )

            metafunc.parametrize(
                "flowsheet_interface",
                paramsets,
                # with indirect=True the value will be "routed" through the fixture of the same name
                # before being passed to the test, so that the original value (the module name)
                # is converted to a FlowsheetInterface instance
                indirect=True,
                scope="class",
            )

    @pytest.fixture(scope="class")
    def flowsheet_interface(self, request: pytest.FixtureRequest) -> FlowsheetInterface:
        module_name: str = request.param
        interface = FlowsheetInterface.from_module(module_name)
        return interface

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        def _sort_key(item: pytest.Item):
            marker = item.get_closest_marker(self._marker_name)
            if marker is None:
                return tuple()
            return marker.args

        items.sort(key=_sort_key)


plugin = IdaesFlowsheetsPlugin()
