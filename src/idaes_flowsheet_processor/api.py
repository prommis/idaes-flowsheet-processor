#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
"""
Simple flowsheet interface API
"""

__author__ = "Dan Gunter"

# stdlib
import abc
import importlib
from collections import namedtuple
from csv import reader, writer
from enum import Enum
from io import TextIOBase

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files
import inspect
import logging
from math import ceil
from operator import itemgetter
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, List, Optional, Dict, Tuple, Union, TypeVar
from types import ModuleType


try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

# third-party
from idaes.core.util.model_statistics import degrees_of_freedom
from pydantic import BaseModel, Field, field_validator, ValidationInfo, ConfigDict
import pyomo.environ as pyo
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from IPython.core.display import HTML

from .util import ShortPrefix

#: Forward-reference to a FlowsheetInterface type, used in
#: :meth:`FlowsheetInterface.find`
FSI = TypeVar("FSI", bound="FlowsheetInterface")


_log = logging.getLogger("idaes." + __name__)
_log.setLevel(logging.INFO)


class UnsupportedObjType(TypeError):
    def __init__(self, obj: Any, supported=None):
        msg = f"Object '{obj}' of type '{type(obj)}' is not supported."
        if supported is not None:
            msg += f"\nSupported: {supported}"
        super().__init__(msg)
        self.obj = obj
        self.supported = supported


_SupportedObjType = Union[
    pyo.Var,
    pyo.Expression,
    pyo.Param,
]
"Used for type hints and as a shorthand in error messages (i.e. not for runtime checks)"


def ensure_supported(obj: object) -> None:
    """Raise UnsupportedObjType if object type is not supported as an input/output"""
    supported = True
    try:
        if not (
            obj.is_variable_type()
            or obj.is_expression_type()
            or obj.is_parameter_type()
        ):
            supported = False
    except AttributeError:
        supported = False
    if not supported:
        raise UnsupportedObjType(obj, supported=_SupportedObjType)


class ModelExport(BaseModel):
    """A variable, expression, or parameter."""

    # TODO: if Optional[_SupportedObjType] is used for the `obj` type hint,
    # pydantic will run the runtime instance check which is not what we want
    # (as we want/need to use the pyomo is_xxx_type() methods instead)
    # so we're using Optional[object] unless we find a way to tell pydantic to skip this check
    # inputs
    obj: Optional[object] = Field(default=None, exclude=True)
    name: str = ""
    value: float = 0.0
    ui_units: object = Field(default=None, exclude=True)
    display_units: str = ""
    rounding: float = 0
    description: str = ""
    is_input: bool = True
    is_output: bool = True
    is_readonly: Union[None, bool] = Field(default=None, validate_default=True)
    input_category: Optional[str] = None
    output_category: Optional[str] = None
    chart_type: Optional[str] = None
    chart_group: Optional[str] = None
    # computed
    obj_key: Union[None, str] = Field(default=None, validate_default=True)
    fixed: bool = True
    lb: Union[None, float] = 0.0
    ub: Union[None, float] = 0.0
    num_samples: int = 2
    has_bounds: bool = True
    is_sweep: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("obj")
    @classmethod
    def validate_obj(cls, v: object) -> object:
        if v is not None:
            ensure_supported(v)
        return v

    # Get value from object
    @field_validator("value")
    @classmethod
    def validate_value(cls, v: float, info: ValidationInfo) -> float:
        if info.data.get("obj", None) is None:
            return v
        obj = info.data["obj"]
        ensure_supported(obj)
        return pyo.value(obj)

    # Derive display_units from ui_units
    @field_validator("display_units")
    @classmethod
    def validate_units(cls, v: str, info: ValidationInfo) -> str:
        if not v:
            u = info.data.get("ui_units", pyo.units.dimensionless)
            v = str(pyo.units.get_units(u))
        return v

    # set name dynamically from object
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str, info: ValidationInfo) -> str:
        if not v:
            obj = info.data
            # if initializing from a dict, skip this
            if not isinstance(obj, dict):
                ensure_supported(obj)
                v = getattr(obj, "name", "unknown")
        return v

    @field_validator("is_readonly")
    @classmethod
    def set_readonly_default(cls, v: Optional[bool], info: ValidationInfo) -> bool:
        if v is None:
            v = True
            obj = info.data["obj"]
            ensure_supported(obj)
            if obj.is_variable_type() or (
                obj.is_parameter_type() and obj.parent_component().mutable
            ):
                v = False
        return v

    @field_validator("obj_key")
    @classmethod
    def set_obj_key_default(cls, v: Optional[str], info: ValidationInfo) -> str:
        if v is None:
            obj = info.data["obj"]
            ensure_supported(obj)
            v = str(obj)
        return v


class KPI(BaseModel):
    """Key Performance Indicator"""

    is_table: bool
    has_total: bool
    name: str
    title: str
    units: List[str] = []
    values: List[float] = []
    labels: List[str] = []
    xlab: str = ""
    ylab: str = ""
    total: float = 0.0
    total_label: str = ""


class ModelOption(BaseModel):
    """An option for building/running the model."""

    name: str
    category: str = "Build Options"
    display_name: Union[None, str] = Field(default=None, validate_default=True)
    description: Union[None, str] = Field(default=None, validate_default=True)
    display_values: List[Any] = []
    values_allowed: Union[str, List[Any]]
    min_val: Union[None, int, float] = None
    max_val: Union[None, int, float] = None
    value: Any = None

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, v: Optional[str], info: ValidationInfo) -> str:
        if v is None:
            v = info.data.get("name")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: Optional[str], info: ValidationInfo) -> str:
        if v is None:
            v = info.data.get("display_name")
        return v

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: Any, info: ValidationInfo) -> Any:
        allowed = info.data.get("values_allowed", None)
        # check if values allowed is int or float and ensure valid value
        if allowed == "int":
            if isinstance(v, int):
                min_val = info.data.get("min_val", float("-inf"))
                max_val = info.data.get("max_val", float("-inf"))
                if v >= min_val and v <= max_val:
                    return v
                else:
                    raise ValueError(
                        f"'value' ({v}) not within expected range of [{min_val}-{max_val}]"
                    )
            else:
                raise ValueError(f"'value' ({v}) not a valid integer")
        elif allowed == "float":
            if isinstance(v, int) or isinstance(v, float):
                min_val = info.data.get("min_val", float("-inf"))
                max_val = info.data.get("max_val", float("-inf"))
                if v >= min_val and v <= max_val:
                    return v
                else:
                    raise ValueError(
                        f"'value' ({v}) not within expected range of [{min_val}-{max_val}]"
                    )
            else:
                raise ValueError(f"'value' ({v}) not a valid float")
        # check if values allowed is string
        elif allowed == "string":
            if isinstance(v, str):
                return v
            else:
                raise ValueError(f"'value' ({v}) not a valid string")
        # values_allowed is a list. make sure v is in the list of values allowed
        elif isinstance(allowed, list):
            if v in allowed:
                return v
            else:
                raise ValueError(f"'value' ({v}) not in allowed values: {allowed}")
        else:
            raise ValueError(
                f"{allowed} does not match the following criteria for values_allowed: must be either a list of possible values, or one of 'string', 'int', 'float'."
            )


class FlowsheetExport(BaseModel):
    """A flowsheet and its contained exported model objects."""

    m: object = Field(default=None, exclude=True)
    obj: object = Field(default=None, exclude=True)
    name: Union[None, str] = Field(default="", validate_default=True)
    description: Union[None, str] = Field(default="", validate_default=True)
    exports: Dict[str, ModelExport] = {}
    kpis: Dict[str, KPI] = {}
    kpi_order: list[str] = []
    kpi_options: Dict = {}
    kpi_figures: Dict = {}
    version: int = 2
    requires_idaes_solver: bool = False
    dof: int = 0
    sweep_results: Union[None, dict] = {}
    build_options: Dict[str, ModelOption] = {}

    # set name dynamically from object
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str, info: ValidationInfo) -> str:
        if not v:
            try:
                v = info.data["obj"].name
            except (KeyError, AttributeError):
                pass
            if not v:
                v = "default"
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str, info: ValidationInfo) -> str:
        if not v:
            try:
                v = info.data["obj"].doc
            except (KeyError, AttributeError):
                v = f"{info.data['name']} flowsheet"
        return v

    def add(
        self, *args: object, data: Union[dict, "ModelExport"] = None, **kwargs: object
    ) -> object:
        """Add a new variable (or other model object).

        There are a few different ways of invoking this function. Users will
        typically use this form::

            add(obj=<pyomo object>, name="My value name", ..etc..)

        where the keywords after `obj` match the non-computed names in :class:`ModelExport`.

        If these same name/value pairs are already in a dictionary, this form is more
        convenient::

            add(data=my_dict_of_name_value_pairs)

        If you have an existing ModelExport object, you can add it more directly with::

            add(my_object)
            # -- OR --
            add(data=my_object)


        Args:
            *args: If present, should be a single non-named argument, which is a
                 ModelExport object. Create by adding it.
            data: If present, create from this argument. If it's a dict, create from
                 its values just as from the kwargs. Otherwise, it should be a
                 ModelExport object, and create by adding it.
            kwargs: Name/value pairs to create a ModelExport object.
                    Accepted names and default values are in the ModelExport.

        Raises:
            KeyError: If the name of the Pyomo object is the same as an existing one,
                i.e. refuse to overwrite.
        """
        if len(args) > 1:
            raise ValueError(f"At most one non-keyword arg allowed. Got: {args}")
        if len(args) == 1:
            model_export = args[0]
        elif data is None:
            _log.debug(f"Create ModelExport from args: {kwargs}")
            model_export = ModelExport.model_validate(kwargs)
        else:
            if isinstance(data, dict):
                model_export = ModelExport.model_validate(data)
            else:
                model_export = data
        key = model_export.obj_key
        if key in self.exports:
            raise KeyError(
                f"Adding ModelExport object failed: duplicate key '{key}' (model_export={model_export})"
            )
        if _log.isEnabledFor(logging.DEBUG):  # skip except in debug mode
            _log.debug(
                f"Adding ModelExport object with key={key}: {model_export.model_dump()}"
            )
        self.exports[key] = model_export
        return model_export

    def clear_kpis(self) -> None:
        self.kpis = {}
        self.kpi_order = []
        self.kpi_options = {}

    def add_kpi_values(
        self,
        name: str,
        values: List[float],
        labels: List[str],
        units: Optional[List[str]],
        title: str = "",
    ) -> None:
        """Add a Key Performance Indicator (KPI) for a table of names and values.

        Args:
            name: Name of the KPI
            values: Numeric values
            labels: Labels corresponding to values
            title: Overall description
            units: Descriptive units for the values (optional)
        """
        kpi = KPI(
            is_table=True,
            has_total=False,
            name=name,
            title=title,
            values=values,
            labels=labels,
            units=units,
        )
        self.kpis[name] = kpi
        self.kpi_order.append(name)

    def _add_kpi_vector(self, name, kwargs):
        kpi = KPI(name=name, **kwargs)
        self.kpis[name] = kpi
        self.kpi_order.append(name)

    def add_kpi_barchart(
        self,
        name: str,
        values: List[float],
        labels: List[str],
        title: str,
        xlab: Optional[str] = None,
        ylab: Optional[str] = None,
        units: SyntaxWarning = "none",
    ) -> None:
        """Add a KeyPerformance Indicator (KPI) vector for a barchart.

        Args:
            name: Name of the KPI
            values: Numeric values
            labels: Labels corresponding to values
            title: Chart title
            xlab: Label for x axis
            ylab: Label for y axis
            units: Units for the values, e.g., "%" if they are percentages adding to 100
        """
        self._add_kpi_vector(
            name,
            dict(
                values=values,
                labels=labels,
                title=title,
                xlab=xlab,
                ylab=ylab,
                units=[units],
                has_total=False,
                is_table=False,
            ),
        )

    def add_kpi_total(
        self,
        name: str,
        values: List[float],
        labels: List[str],
        title: str,
        total_label: str,
        units: str = "%",
    ) -> None:
        """Add a Key Performance Indicator (KPI) vector with multiple values and a total

        Args:
            name: Name of the KPI
            values: Numeric values
            labels: Labels corresponding to values
            total_label: Label for the 'total' to which the values sum.
            units: Units for the values, e.g., "%" if they are percentages adding to 100
        """
        total = sum(values)
        self._add_kpi_vector(
            name,
            dict(
                values=values,
                labels=labels,
                title=title,
                units=[units],
                total=total,
                has_total=True,
                is_table=False,
                total_label=total_label,
            ),
        )

    def set_kpi_default_options(self, **options: object) -> None:
        self.kpi_options = options

    def from_csv(self, file: Union[str, Path], flowsheet: object) -> int:
        """Load multiple exports from the given CSV file.

        CSV file format rules:

            * Always use a header row. The names are case-insensitive, order is
              not important. The 'name', 'obj', and 'ui_units' columns are required.
            * Columns names should match the non-computed names in :class:`ModelExport`.
              See `.add()` for a list.
            * The object to export should be in a column named 'obj', prefixed with 'fs.'
            * For units, use Pyomo units module as 'units', e.g., 'mg/L' is `units.mg / units.L`

        For example::

            name,obj,description,ui_units,display_units,rounding,is_input,input_category,is_output,output_category
            Leach liquid feed rate,fs.leach_liquid_feed.flow_vol[0],Leach liquid feed volumetric flow rate,units.L/units.hour,L/h,2,TRUE,Liquid feed,FALSE,
            Leach liquid feed H,"fs.leach_liquid_feed.conc_mass_comp[0,'H']",Leach liquid feed hydrogen mass composition,units.mg/units.L,mg/L,3,TRUE,Liquid feed,FALSE,
            .......etc.......

        Args:
            file: Filename or path. If not an absolute path, start from the
                  directory of the caller's file.
            flowsheet: Flowsheet used to evaluate the exported objects.

        Returns:
            int: Number of exports added

        Raises:
            IOError: if input file doesn't exist
            ValueError: Invalid data in input file (error message will have details)
        """
        _log.debug(f"exports.add: from csv filename={file}")

        # compute path
        path = Path(file) if not isinstance(file, Path) else file
        if path.is_absolute():
            _log.debug(
                f"Reading CSV data for interface exports from " f"absolute path: {path}"
            )
            text = open(path, "r", encoding="utf-8").read()
        else:
            caller = inspect.getouterframes(inspect.currentframe())[1]
            caller_mod = inspect.getmodule(caller.frame).__name__
            if "." not in caller_mod:  # not in a package
                path = Path(caller.filename).parent / file
                text = open(path, "r", encoding="utf-8").read()
            else:
                caller_pkg = ".".join(caller_mod.split(".")[:-1])  # strip module
                _log.debug(
                    f"Reading CSV data for interface exports from: "
                    f"file={path}, module={caller_mod}, package={caller_pkg}"
                )
                try:
                    text = files(caller_pkg).joinpath(path).read_text()
                except Exception as err:
                    raise IOError(
                        f"Could not find CSV file '{path}' relative to file "
                        f"calling .add() in '{caller_mod}': {err}"
                    )

        # process CSV file
        rows = reader(re.split(r"\r?\n", text))
        # read and pre-process the header row
        raw_header = next(rows)
        header = [s.strip().lower() for s in raw_header]
        for req in "name", "obj", "ui_units":
            if req not in header:
                raise ValueError(
                    f"Bad CSV header: '{req}' column is required. data=" f"{header}"
                )
        num = 0
        for row in rows:
            if len(row) == 0:
                continue
            # build raw dict from values and header
            data = {k: v for k, v in zip(header, row)}
            # evaluate the object in the flowsheet
            try:
                data["obj"] = eval(data["obj"], {"fs": flowsheet})
            except Exception as err:
                raise ValueError(f"Cannot find object in flowsheet: {data['obj']}")
            # evaluate the units
            norm_units = data["ui_units"].strip()
            if norm_units in ("", "none", "-"):
                data["ui_units"] = pyo.units.dimensionless
            else:
                try:
                    data["ui_units"] = eval(norm_units, {"units": pyo.units})
                except Exception as err:
                    raise ValueError(f"Bad units '{norm_units}': {err}")
            # process boolean values (starting with 'is_')
            for k in data:
                if k.startswith("is_"):
                    v = data[k].lower()
                    if v == "true":
                        data[k] = True
                    elif v == "false":
                        data[k] = False
                    else:
                        raise ValueError(
                            f"Bad value '{data[k]}' "
                            f"for boolean argument '{k}': "
                            f"must be 'true' or 'false' "
                            f"(case-insensitive)"
                        )
            # add parsed export
            self.add(data=data)
            num += 1

        return num

    def to_csv(self, output: Union[TextIOBase, Path, str] = None) -> int:
        """Write wrapped objects as CSV.

        Args:
            output: Where to write CSV file. Can be a stream, path, or filename.

        Returns:
            Number of objects written into file.

        Raises:
            IOError: If path is given, and not writable
        """
        # open file for writing
        if isinstance(output, TextIOBase):
            output_file = output
        else:
            p = Path(output)
            output_file = p.open("w")

        # initialize
        csv_output_file = writer(output_file)

        # write header row
        obj = next(iter(self.exports.values()))
        values = ["obj", "ui_units"]
        col_idx_map = {}
        for i, field_name in enumerate(obj.model_dump()):
            # add to mapping of field name to column number
            col_idx_map[field_name] = i + 2
            # add column name
            values.append(field_name)
        csv_output_file.writerow(values)
        ncol = len(values)

        # write a row for each object
        num = 0
        for key, obj in self.exports.items():
            # initialize values list
            #   first 2 column values are object name and units
            obj_name = self._massage_object_name(key)
            units_str = self._massage_ui_units(str(obj.ui_units))
            values = [obj_name, units_str] + [""] * (ncol - 2)
            # add columns
            for field_name, field_value in obj.model_dump().items():
                values[col_idx_map[field_name]] = field_value
            # write row
            csv_output_file.writerow(values)
            num += 1

        return num

    @staticmethod
    def _massage_object_name(s: str) -> str:
        s1 = re.sub(r"\[([^]]*)\]", r"['\1']", s)  # quote everything in [brackets]
        s2 = re.sub(r"\['([0-9.]+)'\]", r"[\1]", s1)  # unquote [0.0] numbers
        return s2

    @staticmethod
    def _massage_ui_units(s: str) -> str:
        if s == "dimensionless":
            return ""
        return s

    def add_option(self, name: str, **kwargs: object) -> ModelOption:
        """Add an 'option' to the flowsheet that can be displayed and manipulated
        from the UI.

        Constructs a :class:`ModelOption` instance with provided args and adds it to
        the dict of options, keyed by its `name`.

        Args:
            name: Name of option (internal, for accessing the option)
            kwargs: Fields of :class:`ModelOption`
        """
        option = ModelOption(name=name, **kwargs)
        self.build_options[name] = option
        return option


class Actions(str, Enum):
    """Known actions that can be run.
    Actions that users should not run directly (unless they know what they are
    doing) are prefixed with an underscore.
    """

    build = "build"
    solve = "solve"
    export = "_export"
    diagram = "diagram"
    initialize = "initialize"
    kpis = "kpis"


class FlowsheetCategory(str, Enum):
    """Flowsheet Categories"""

    wastewater = "Wasterwater Recovery"
    desalination = "Desalination"


class FlowsheetReport(abc.ABC):
    """Abstract base class for flowsheet reports.

    This class defines the interface for generating reports from a flowsheet.
    Subclasses should implement the `to_html` method to generate the report in HTML format.
    """

    @abc.abstractmethod
    def to_html(self, **kwargs: object) -> str:
        """Return report as an HTML string."""
        pass


class FlowsheetInterface:
    """Interface between users, UI developers, and flowsheet models."""

    #: Function to look for in modules. See :meth:`find`.
    UI_HOOK = "export_to_ui"

    _KPI_REPORT = "Key Performance Indicators"

    #: Type of item in list ``MissingObjectError.missing``.
    #: ``key`` is the unique key assigned to the variable,
    #: ``name`` is the variable name in the flowsheet
    MissingObject = namedtuple("MissingObject", "key name")

    class MissingObjectError(Exception):
        """Error returned if data in `load` refers to a variable not found in the
        target object.

        Use the `.missing` attribute of the error object to get the list  of
        MissingObjects.
        """

        def __init__(self, missing):
            num = len(missing)
            plural = "" if num == 1 else "s"
            things = [f"{m[1]}" for m in missing]
            super().__init__(
                f"{num} object{plural} not found in the model: {', '.join(things)}"
            )
            self.missing = [
                FlowsheetInterface.MissingObject(key=m[0], name=m[1]) for m in missing
            ]

    def __init__(
        self,
        fs: Optional[FlowsheetExport] = None,
        do_build: Optional[Callable] = None,
        do_export: Optional[Callable] = None,
        do_solve: Optional[Callable] = None,
        do_initialize: Optional[Callable] = None,
        do_kpis: Optional[Callable] = None,
        get_diagram: Optional[Callable] = None,
        category: Optional["FlowsheetCategory"] = None,
        custom_do_param_sweep_kwargs: Optional[Dict] = None,
        **kwargs: object,
    ) -> None:
        """Constructor.

        Args:
            fs: An existing wrapper to a flowsheet object. If this is not provided,
                then one will be constructed by passing the keyword arguments to
                the built-in pydantic ``model_validate()`` method
                of :class:`FlowsheetExport`.
            do_build: Function to call to build the flowsheet. It should build the
                flowsheet model and return the `FlowsheetBlock`, which is typically
                the `fs` attribute of the model object. **Required**
            do_export: Function to call to export variables after the model is built.
                This will be called automatically by :meth:`build()`. **Required**
            do_solve: Function to solve the model. It should return the result
                that the solver itself returns. **Required**
            do_kpis: Function to set KPIs.
            custom_do_param_sweep_kwargs: Option for setting up parallel solver using
                custom solve function.
            **kwargs: See `fs` arg. If the `fs` arg *is* provided, these are ignored.
        """
        if fs is None:
            self.fs_exp = FlowsheetExport.model_validate(kwargs)
        else:
            self.fs_exp = fs
        self._actions = {}
        for arg, name in (
            (do_export, "export"),
            (do_build, "build"),
            (do_solve, "solve"),
            (do_initialize, "initialize"),
        ):
            if arg:
                if not callable(arg):
                    raise TypeError(f"'do_{name}' argument must be callable")
                self.add_action(getattr(Actions, name), arg)
            elif name == "initialize":
                self.add_action(getattr(Actions, name), arg)
            else:
                raise ValueError(f"'do_{name}' argument is required")
        report_types = []
        # optional kpis
        if callable(do_kpis):
            self.add_action(Actions.kpis, do_kpis)
            report_types.append(self._KPI_REPORT)
        # optional diagram
        if callable(get_diagram):
            self.add_action("diagram", get_diagram)
        else:
            self.add_action("diagram", None)

        self._actions["custom_do_param_sweep_kwargs"] = custom_do_param_sweep_kwargs

        # for selecting report types
        self._report_prefix = ShortPrefix(report_types)

    def build(self, quiet=False, **kwargs: object) -> None:
        """Build flowsheet

        Args:
            quiet: If true, suppress output from the build function
            **kwargs: User-defined values

        Returns:
            None

        Raises:
            RuntimeError: If the build fails
        """
        try:
            if quiet:
                _save_loglevel = _log.getEffectiveLevel()
                _log.setLevel(logging.WARNING)
                idaes_log = logging.getLogger("idaes")
                _save_idaes_loglevel = idaes_log.getEffectiveLevel()
                idaes_log.setLevel(logging.WARNING)
                _save_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")
            try:
                self.run_action(Actions.build, **kwargs)
            except Exception as err:
                raise RuntimeError(f"Building flowsheet: {err}") from err
        finally:
            if quiet:
                sys.stdout = _save_stdout
                _log.setLevel(_save_loglevel)
                idaes_log.setLevel(_save_idaes_loglevel)
        return

    def solve(self, **kwargs: object) -> Any:
        """Solve flowsheet.

        Args:
            **kwargs: User-defined values

        Returns:
            Return value of the underlying solve function

        Raises:
            RuntimeError: if the solver did not terminate in an optimal solution
        """
        try:
            result = self.run_action(Actions.solve, **kwargs)
        except Exception as err:
            raise RuntimeError(f"Solving flowsheet: {err}") from err
        return result

    def get_diagram(self, **kwargs: object) -> Optional[Any]:
        """Return diagram image name.

        Args:
            **kwargs: User-defined values

        Returns:
            Return image file name if get_diagram function is callable. Otherwise, return none
        """
        if self.get_action(Actions.diagram) is not None:
            return self.run_action(Actions.diagram, **kwargs)
        else:
            return None

    def initialize(self, *args: object, **kwargs: object) -> Any:
        """Run initialize function.

        Args:
            **kwargs: User-defined values

        Returns:
            Return value of the underlying initialization function. Otherwise, return none
        """
        if self.get_action(Actions.initialize) is not None:
            try:
                result = self.run_action(Actions.initialize, *args, **kwargs)
            except Exception as err:
                raise RuntimeError(f"Initializing flowsheet: {err}") from err
            return result
        else:
            return None

    def dict(self) -> Dict:
        """Serialize.

        Returns:
            Serialized contained FlowsheetExport object
        """
        return self.fs_exp.model_dump(exclude={"obj"})

    def load(self, data: Dict) -> None:
        """Load values from the data into corresponding variables in this
        instance's FlowsheetObject.

        Args:
            data: The input flowsheet (probably deserialized from JSON)
        """
        u = pyo.units
        fs = FlowsheetExport.model_validate(data)  # new instance from data
        # Set the value for each input variable
        missing = []
        # 'src' is the data source and 'dst' is this flowsheet (destination)
        for key, src in fs.exports.items():
            # get corresponding exported variable
            try:
                dst = self.fs_exp.exports[key]
            except KeyError:
                missing.append((key, src.name))
                continue
            # set value in this flowsheet
            ui_units = dst.ui_units
            if dst.is_input and not dst.is_readonly:
                # only update if value has changed
                if dst.value != src.value:
                    # print(f'changing value for {key} from {dst.value} to {src.value}')
                    # create a Var so Pyomo can do the unit conversion for us
                    tmp = pyo.Var(initialize=src.value, units=ui_units)
                    tmp.construct()
                    # Convert units when setting value in the model
                    new_val = pyo.value(u.convert(tmp, to_units=u.get_units(dst.obj)))
                    # print(f'changing value for {key} from {dst.value} to {new_val}')
                    dst.obj.set_value(new_val)
                    # Don't convert units when setting the exported value
                    dst.value = src.value

                # update other variable properties if changed, not applicable for parameters
                if dst.obj.is_variable_type():
                    if dst.obj.fixed != src.fixed:
                        # print(f'changing fixed for {key} from {dst.obj.fixed} to {src.fixed}')
                        if src.fixed:
                            dst.obj.fix()
                        else:
                            dst.obj.unfix()
                        dst.fixed = src.fixed
                    # update bounds
                    if dst.lb != src.lb:
                        # print(f'changing lb for {key} from {dst.lb} to {src.lb}')
                        if src.lb is None or src.lb == "":
                            dst.obj.setlb(None)
                            dst.lb = None
                        else:
                            tmp = pyo.Var(initialize=src.lb, units=ui_units)
                            tmp.construct()
                            new_lb = pyo.value(
                                u.convert(tmp, to_units=u.get_units(dst.obj))
                            )
                            dst.obj.setlb(new_lb)
                            dst.lb = src.lb
                    if dst.ub != src.ub:
                        # print(f'changing ub for {key} from {dst.ub} to {src.ub}')
                        if src.ub is None or src.ub == "":
                            dst.obj.setub(None)
                            dst.ub = None
                        else:
                            tmp = pyo.Var(initialize=src.ub, units=ui_units)
                            tmp.construct()
                            new_ub = pyo.value(
                                u.convert(tmp, to_units=u.get_units(dst.obj))
                            )
                            # print(f'changing ub for {key} from {dst.obj.ub} to {new_ub}')
                            dst.obj.setub(new_ub)
                            dst.ub = src.ub

                if dst.is_sweep != src.is_sweep:
                    dst.is_sweep = src.is_sweep

                if dst.num_samples != src.num_samples:
                    dst.num_samples = src.num_samples

        # update degrees of freedom (dof)
        self.fs_exp.dof = degrees_of_freedom(self.fs_exp.obj)
        if missing:
            raise self.MissingObjectError(missing)

    def select_option(self, option_name: str, new_option: str) -> None:
        """Update flowsheet with selected option.

        Args:
            data: The input flowsheet
            option_name: Name of selected option

        Returns:
            None
        """

        # fs = FlowsheetExport.model_validate(data)  # new instance from data
        self.fs_exp.build_options[option_name].value = new_option

        # # get function name from model options
        # func_name = self.fs_exp.build_options[option_name].values_allowed[new_option]

        # # add functino name as new build function
        # self.add_action("build", func_name)

    def add_action(self, action_name: str, action_func: Callable) -> None:
        """Add an action for the flowsheet.

        Args:
            action_name: Name of the action to take (see :class:`Actions`)
            action_func: Function to call for the action

        Returns:
            None
        """
        _log.debug(f"adding action: {action_name}")

        def action_wrapper(*args, **kwargs):
            _log.debug(f"running action: {action_name}")
            if action_name == Actions.build:
                # set new model object from return value of build action
                action_result = action_func(**kwargs)
                if action_result is None:
                    raise RuntimeError(
                        f"Flowsheet `{Actions.build}` action failed. "
                        f"See logs for details."
                    )
                self.fs_exp.obj = action_result.fs
                self.fs_exp.m = action_result
                # [re-]create exports (new model object)
                if Actions.export not in self._actions:
                    raise KeyError(
                        "Error in 'build' action: no export action defined. "
                        "Add `do_export=<function>` to FlowsheetInterface "
                        "constructor or call `add_action(Actions.export, <function>)` "
                        "on FlowsheetInterface instance."
                    )
                # clear exports dict, since duplicates not allowed
                self.fs_exp.exports.clear()
                # use get_action() since run_action() will refuse to call it directly
                self.get_action(Actions.export)(
                    exports=self.fs_exp, build_options=self.fs_exp.build_options
                )
                # done
                result = None
            elif action_name == Actions.diagram:
                self._actions[action_name] = action_func
                return
            elif action_name == Actions.initialize:
                _log.debug(f"initializing")
                result = action_func(self.fs_exp.m)
            elif self.fs_exp.obj is None:
                raise RuntimeError(
                    f"Cannot run any flowsheet action (except "
                    f"'{Actions.build}') before flowsheet is built"
                )
            else:
                if action_name == Actions.kpis:
                    result = action_func(exports=self.fs_exp, flowsheet=self.fs_exp.obj)
                else:
                    # all others, call with flowsheet object
                    result = action_func(flowsheet=self.fs_exp.obj, **kwargs)
                if action_name == Actions.solve:
                    _log.debug(f"Solve result: {result}")
                    if result is None:
                        raise RuntimeError("Solver did not return a result")
                    if not pyo.check_optimal_termination(result):
                        raise RuntimeError(f"Solve failed: {result}")
            # Sync model with exported values
            if action_name in (Actions.build, Actions.solve, Actions.initialize):
                self.export_values()
                # (re-)add KPIs if any specified
                if Actions.kpis in self._actions:
                    _log.info("Calculating and exporting KPIs")
                    self.fs_exp.clear_kpis()
                    self.run_action(Actions.kpis, self.fs_exp)
                else:
                    _log.info("No KPIs present")
            return result

        self._actions[action_name] = action_wrapper

    def get_action(self, name: str) -> Optional[Callable]:
        """Get the function for an ``add()``-ed action.

        Args:
            name: Name of the action (see :class:`Actions`)

        Returns:
            Function for this action

        Raises:
            KeyError, if no such action is defined
        """
        return self._actions[name]

    def run_action(self, name: str, *args: object, **kwargs: object) -> Any:
        """Run the named action."""
        func = self.get_action(name)
        if name.startswith("_"):
            raise ValueError(
                f"Refusing to call '{name}' action directly since its "
                f"name begins with an underscore"
            )
        return func(*args, **kwargs)

    def export_values(self) -> None:
        """Copy current values in underlying Pyomo model into exported model.

        Side-effects:
            Attribute ``fs_exp`` is modified.
        """
        _log.info("Exporting values from flowsheet model to UI")
        u = pyo.units
        self.fs_exp.dof = degrees_of_freedom(self.fs_exp.obj)
        for key, mo in self.fs_exp.exports.items():
            mo.value = pyo.value(u.convert(mo.obj, to_units=mo.ui_units))
            # print(f'{key} is being set to: {mo.value}')
            if hasattr(mo.obj, "bounds"):
                # print(f'{key} is being set to: {mo.value} from {mo.obj.value}')
                if mo.obj.ub is None:
                    mo.ub = mo.obj.ub
                else:
                    tmp = pyo.Var(initialize=mo.obj.ub, units=u.get_units(mo.obj))
                    tmp.construct()
                    mo.ub = pyo.value(u.convert(tmp, to_units=mo.ui_units))
                if mo.obj.lb is None:
                    mo.lb = mo.obj.lb
                else:
                    tmp = pyo.Var(initialize=mo.obj.lb, units=u.get_units(mo.obj))
                    tmp.construct()
                    mo.lb = pyo.value(u.convert(tmp, to_units=mo.ui_units))
                mo.fixed = mo.obj.fixed
            else:
                mo.has_bounds = False

    @classmethod
    def from_installed_packages(
        cls, group_name: str = "watertap.flowsheets"
    ) -> Dict[str, "FlowsheetInterface"]:
        """Get all flowsheet interfaces defined as entry points within the Python packages installed in the environment.

        This uses the :func:`importlib.metadata.entry_points` function to fetch the
        list of flowsheets declared as part of a Python package distribution's `entry points <https://docs.python.org/3/library/importlib.metadata.html#entry-points>`_
        under the group ``group_name``.

        To set up a flowsheet interface for discovery, locate your Python package distribution's file (normally
        :file:`setup.py`, :file:`pyproject.toml`, or equivalent) and add an entry in the ``entry_points`` section.

        For example, to add a flowsheet defined in :file:`watertap/flowsheets/flowsheets/my_flowsheet.py`
        so that it can be discovered with the name ``my_flowsheet`` wherever the ``watertap`` package is installed,
        the following should be added to WaterTAP's :file:`setup.py`::

           setup(
               name="watertap",
               # other setup() sections
               entry_points={
                   "watertap.flowsheets": [
                        # other flowsheet entry points
                        "my_flowsheet = watertap.flowsheets.flowsheets.my_flowsheet",
                   ]
               }
           )

        Args:
            group_name: The entry_points group from which the flowsheet interface modules will be populated.

        Returns:
            Mapping with keys the module names and values FlowsheetInterface objects
        """
        eps = metadata.entry_points()
        try:
            # this happens for Python 3.7 (via importlib_metadata) and Python 3.10+
            entry_points = list(eps.select(group=group_name))
        except AttributeError:
            # this will happen on Python 3.8 and 3.9, where entry_points() has dict-like group selection
            entry_points = list(eps[group_name])

        if not entry_points:
            _log.error(f"No interfaces found for entry points group: {group_name}")
            return {}

        interfaces = {}
        _log.debug(f"Loading {len(entry_points)} entry points")
        for ep in entry_points:
            _log.debug(f"ep = {ep}")
            module_name = ep.value
            try:
                module = ep.load()
            except ImportError as err:
                _log.error(f"Cannot import module '{module_name}': {err}")
                continue
            interface = cls.from_module(module)
            if interface:
                interfaces[module_name] = interface

        return interfaces

    @classmethod
    def from_module(
        cls, module: Union[str, ModuleType]
    ) -> Optional["FlowsheetInterface"]:
        """Get a a flowsheet interface for module.

        Args:
            module: The module

        Returns:
            A flowsheet interface or None if it failed
        """
        if not isinstance(module, ModuleType):
            module = importlib.import_module(module)

        # Get function that creates the FlowsheetInterface
        func = getattr(module, cls.UI_HOOK, None)
        if func is None:
            _log.warning(
                f"Interface for module '{module}' is missing UI hook function: "
                f"{cls.UI_HOOK}()"
            )
            return None
        # Call the function that creates the FlowsheetInterface
        try:
            interface = func()
        except Exception as err:
            _log.error(
                f"Cannot get FlowsheetInterface object for module '{module}': {err}"
            )
            return None
        # Return created FlowsheetInterface
        return interface

    def report(self, rtype=_KPI_REPORT, **kwargs: object) -> FlowsheetReport:
        """Generate and return a report for the flowsheet.



        Args:
            report_type: Type of report to generate (any unique prefix of the report type).
            kwargs: Additional keywords passed to the given report type.

        Raises:
            ValueError: If a report argument is invalid

        Returns:
            FlowsheetReport: A flowsheet report, which will display automatically in Jupyter Notebooks.
        """
        m = self._report_prefix.match(rtype)
        if m is None:
            cbe = self._report_prefix.could_be(rtype)
            if cbe:
                raise ValueError(
                    f"Report type prefix '{rtype}' has multiple matches: {', '.join(cbe)}"
                )
            else:
                raise ValueError(
                    f"Unknown report type '{rtype}'. "
                    f"Valid types are: {', '.join(self._report_prefix.words)}"
                )
        if m == self._KPI_REPORT:
            return FlowsheetKPIReport(self.fs_exp, **kwargs)
        else:
            # should never happen, since we already checked the prefix
            raise RuntimeError("Unknown report type: {report_type}")


class _ChartTypes(Enum):
    waffle = 1
    donut = 2


WAFFLE = _ChartTypes.waffle
DONUT = _ChartTypes.donut


class FlowsheetKPIReport(FlowsheetReport):
    """Report of the Key Performance Indicators (KPIs) for a flowsheet.

    The specification of the report is extracted from the FlowsheetExport object
    that is passed to the class constructor.
    """

    # Some settings for HTML display
    VALUE_FONT_NAME_SIZE = "150%"
    VALUE_FONT_VAL_SIZE = "180%"
    VALUE_FONT_NAME_COLOR = "#666"
    VALUE_FONT_VAL_COLOR = "#66F"
    VALUE_SEP_WIDTH = "7px"

    def __init__(
        self,
        flowsheet_export: FlowsheetExport,
        total_type: Optional[_ChartTypes] = None,
        bgcolor: str = "#ffffff",
        **kwargs: object,
    ) -> None:
        """Constructor.

        Args:
            flowsheet_export (FlowsheetExport): Exported flowsheet
            total_type (_ChartTypes, optional): How to represent 'total' information. Defaults to WAFFLE.
            bgcolor: Plot and overall ('paper') background color, as a valid color string
            kwargs: Additional key/value pairs passed to the various `create_*` methods
        """
        self._kpis = flowsheet_export.kpis
        self._kpi_ord = flowsheet_export.kpi_order
        self._kpi_opt = flowsheet_export.kpi_options
        self._init_layout = None
        if "layout" in kwargs:
            self._init_layout = kwargs["layout"]
            del kwargs["layout"]
        self._init_options = kwargs
        if total_type:
            self._init_options["total_type"] = total_type
        self._bgcolor = bgcolor

    def to_html(self, layout: Optional[Any] = None, **kwargs: object) -> str:
        """Build the report and return as a complete <HTML> element.

        Args:
            layout: Layout specification. See :class:`Layout` for format. If not given, lay out in one column.
            kwargs: Options for the `create_*` methods.

        Returns:
            HTML for the report
        """
        figures = self.get_kpi_figures(**kwargs)
        figures_html = {key: val.to_html() for key, val in figures.items()}
        # if no provided layout, put each item in its own row
        if layout is None and self._init_layout is not None:
            layout = self._init_layout
        spec = layout if layout else [[key] for key in self._kpi_ord]
        _log.debug(f"layout spec={spec}")
        # perform the layout
        layout_obj = Layout(spec, figures_html)
        body, css = layout_obj.body, layout_obj.css

        # return HTML
        report_css = f"body {{background-color: '{self._bgcolor}';}}"
        html_head = f"<head><style>{report_css}\n{css}</style></head>"
        html_body = f"<body>{body}</body>"
        return f"<html>{html_head}{html_body}</html>"

    _repr_html_ = to_html  # display automatically in Jupyter Notebooks

    def get_kpi_figures(self, **kwargs: object) -> dict[str, go.Figure]:
        """ "Get figures for each KPI.

        Args:
            kwargs: Options for the `create_*` methods.

        Returns:

        """
        # combine options from user settings, constructor, and this method
        options = {}
        options.update(self._kpi_opt)
        options.update(self._init_options)
        options.update(kwargs)
        # special processing for total_type, if given
        if "total_type" in options:
            options["total_type"] = self._preprocess_total_type(options["total_type"])
        # get figure for each KPI
        return {
            key: self._kpi_figure(kpi, **options) for key, kpi in self._kpis.items()
        }

    @staticmethod
    def _preprocess_total_type(v: object) -> _ChartTypes:
        if isinstance(v, _ChartTypes):
            pass  # do nothing
        elif isinstance(v, str):
            s = v.lower().strip()
            try:
                v = _ChartTypes[s]
            except KeyError:
                raise ValueError(f"Unknown total_type: {v}")
        else:
            raise ValueError(
                f"total_type value must be enumeration or string, got {type(v)}"
            )
        return v

    @classmethod
    def _kpi_figure(cls, kpi: KPI, **options: object) -> go.Figure:
        """Get Figure for one KPI.

        Args:
            options: Keyword options for the `create_*` method called
                     to create this KPI.
        """
        if not kpi.is_table:
            if kpi.has_total:
                item = cls.create_kpi_total(kpi, **options)
            else:
                item = cls.create_kpi_barchart(kpi, **options)
        else:
            item = cls.create_kpi_values(kpi, **options)
        return item

    @classmethod
    def create_kpi_values(
        cls,
        kpi,
        font_size: int = 24,
        width: int = 800,
        margin: Optional[int] = None,
        **ignore: object,
    ) -> go.Figure:
        """Create a table with values for each KPI.

        Args:
            kpi
            font_size: Font size for the table, in points
            width: Width of the table in pixels
            margin: Margin around the table, as a dict with keys 't', 'b', 'l', 'r'. Values in pixels.

        Returns:
            Plotly Figure object with values in a table
        """
        row_height = int(font_size * 1.5)
        colors = ["#9999ff", "black"]
        if kpi.units is None:
            values = kpi.values
        else:
            values = [f"{v} {u}" for u, v in zip(kpi.units, kpi.values)]
        value_columns = [kpi.labels, values]
        data = [
            {
                "type": "table",
                "header": {
                    "values": [],
                    "fill": {"color": "rgba(0,0,0,0)"},
                    "height": 0,
                },
                "cells": {
                    "values": value_columns,
                    "align": "center",
                    "height": row_height,
                    "font": {"family": "Arial", "size": font_size, "color": colors},
                    "fill": {"color": "white"},
                },
            }
        ]
        layout = {
            "width": width,
            "margin": margin or {"t": 20, "b": 20, "l": 20},
            "height": 64 + row_height * len(kpi.labels),
        }
        return go.Figure(data, layout=layout)

    @classmethod
    def create_kpi_barchart(cls, kpi: KPI, **ignore: object) -> go.Figure:
        """Create a barchart from a vector of values

        Args:
            kpi: Key performance indicator

        Returns:
            Plotly Figure object with the bar chart
        """
        df = pd.DataFrame(dict(y=kpi.values, x=kpi.labels))
        u = f" ({kpi.units[0]})"
        fig = px.bar(
            df,
            x="x",
            y="y",
            labels={"x": kpi.xlab, "y": kpi.ylab + u},
            title=kpi.title,
        )
        return fig

    @classmethod
    def create_kpi_total(
        cls, kpi: KPI, total_type: _ChartTypes = WAFFLE, **ignore: object
    ) -> go.Figure:
        """Create diagram for a vector that should be represented as parts of a total.
        This will be either a pie (donut) chart or waffle chart.

        Args:
            kpi: Key performance indicator

        Returns:
            Plotly Figure object with the pie or waffle chart
        """
        kpi.units = kpi.units[0]
        if total_type == DONUT:
            fig = px.pie(names=kpi.labels, values=kpi.values, hole=0.5, title=kpi.title)
        else:  # total_type == _ChartTypes.waffle
            fig = cls._waffle_chart(kpi)
        return fig

    @classmethod
    def _waffle_chart(cls, kpi: KPI) -> Any:
        """Create a waffle chart using the imshow() plot."""
        if kpi.total == 0:
            raise ValueError(
                f"KPI '{kpi.title}' has total of 0, cannot create waffle chart"
            )
        # import here to avoid circular import
        # sort (value, label) pairs by value
        val_lab = list(zip(kpi.values, kpi.labels))
        val_lab.sort(key=itemgetter(0))
        n = len(val_lab)
        ttl = kpi.total
        # convert colors to (r,g,b,a) tuples
        col_hex = px.colors.qualitative.Alphabet
        col_rgba = [
            (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16), 255) for s in col_hex
        ]
        # calculate how many boxes to draw for each value
        # Note: this may be more than the grid size due to ceil() function
        width = 20
        grid_sz = width * width
        val_grid = [int(ceil(val_lab[i][0] / ttl * grid_sz)) for i in range(n)]
        vec = []
        for i in range(n):
            for j in range(val_grid[i]):
                vec.append(col_rgba[i])
        # reshape to width x width grid
        nv = len(vec)
        # add empty squares at end of last row
        rows = int(ceil(nv / width))
        for i in range(rows * width - nv):
            vec.append((255, 255, 255, 255))
        arr = np.array(vec, dtype=np.uint8)
        z = arr.reshape(width, rows, 4)
        # draw lines between squares
        ncol, nrow = z.shape[0], z.shape[1]
        fig = px.imshow(z, title=kpi.title)
        line_style = {"color": "white", "width": 2}
        for i in range(ncol):
            fig.add_shape(
                type="line",
                x0=i + 0.5,
                y0=-0.5,
                x1=i + 0.5,
                y1=ncol - 0.5,
                line=line_style,
            )
        for i in range(nrow - 1):
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=i + 0.5,
                x1=nrow - 0.5,
                y1=i + 0.5,
                line=line_style,
            )
        # add fake traces to create a legend
        scatter_kw = dict(x=[None], y=[None], mode="markers")
        for i, (val, lab) in enumerate(val_lab):
            col = col_hex[i]
            mark = dict(size=7, color=col, symbol="square")
            grid_pct = 100 / grid_sz
            if val / ttl < 1 / grid_sz:
                pct_str = f"<{grid_pct:.2f}%"
            else:
                pct = int(val_grid[i] * grid_pct)
                pct_str = f"{pct:2d}%"
            short_val = f"{val:0.3g}"
            name = f"{lab:2s}  {pct_str} {short_val}"
            fig.add_trace(go.Scatter(name=name, marker=mark, **scatter_kw))
        # style plot
        fig.update_layout(
            width=800,
            xaxis_visible=False,
            yaxis_visible=False,
            showlegend=True,
            font_family="Courier New",  # fixed-width
            font_color="#333",
            title_font_family="Arial",
            legend_title_font_color="#666",
        )
        return fig


class Layout:
    """Translate a simple layout specification into an HTML flex layout.

    The specification takes the form of a (possibly nested) list of
    names of the KPIs to display, where the outer list is a single
    column and the next level are rows, then columns within each row, etc.

    Some examples:

    * `[ "kpi_one", "kpi_two", "kpi_three" ]` will display 4 rows, each the full column width.
    * `[["kpi_one", "kpi_two", "kpi_three"]]` will display 1 row with items laid out horizontally
    * `[["kpi_one", "kpi_two"], ["kpi_three", "kpi_four"]]` will display a 2x2 grid
    """

    FLOW_COL, FLOW_ROW = "grid_column", "grid_row"

    def __init__(self, spec, kpis: dict[str, str]):
        """Create new layout.

        Args:
            spec: Layout specification (see class docstring)
            kpis: Mapping of KPI names to HTML. These names should match names in `spec`.
        """
        self._kpis = kpis
        self._divs = self._to_divs(spec, self.FLOW_COL)

    @property
    def body(self):
        return "".join(self._divs)

    @property
    def css(self):
        rules = (
            ".grid_row {display: flex; flex-direction: row;}",
            ".grid_column {display: flex; flex-direction: column}",
        )
        return "\n".join(rules)

    def _toggle_flow(self, flow):
        return self.FLOW_COL if flow == self.FLOW_ROW else self.FLOW_ROW

    def _to_divs(self, item, flow):
        divs = [f"<div class='{flow}'>"]
        if isinstance(item, list):
            next_flow = self._toggle_flow(flow)
            for child in item:
                divs.extend(self._to_divs(child, next_flow))
        else:
            kpi_html = self._kpis[item]
            divs.append(kpi_html)
        divs.append("</div>")
        return divs
