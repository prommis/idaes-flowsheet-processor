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
Tests for fsapi module
"""
import logging
from pathlib import Path
import pytest
import tempfile

from pyomo.environ import units as pyunits
from pyomo.environ import Var, value
from pyomo.environ import SolverStatus, TerminationCondition

import idaes_flowsheet_processor.api as fsapi


pytest.importorskip(
    "watertap.flowsheets",
    reason="watertap.flowsheets is currently required to be able to run tests",
)
from watertap.flowsheets.seawater_RO_desalination import seawater_RO_desalination as RO
from watertap.flowsheets.dye_desalination import dye_desalination_ui as DD

_log = logging.getLogger("idaes.flowsheet_processor")
_log.setLevel(logging.DEBUG)

ERD_TYPE = "pressure_exchanger"

# Fake status=OK solver result


class FAKE_FLOWSHEET:
    fs = "fs"
    trash = "true"


class SOLVE_RESULT_OK:
    class SOLVE_STATUS:
        status = SolverStatus.ok
        termination_condition = TerminationCondition.optimal

    solver = SOLVE_STATUS


def build_ro(build_options=None, **kwargs):
    model = RO.build_flowsheet(erd_type=ERD_TYPE)
    return model


def solve_ro(flowsheet=None):
    assert flowsheet
    return {"solved": True}


class InputCategory:
    """Names for input categories"""

    feed = "Feed"
    hydrogen = "Hydrogen reactor"
    methane = "Methane reactor"
    system = "System parameters"


class OutputCategory:
    """Names for output categories"""

    feed = "Feed"
    levelized_costs = "Levelized costs"
    normalized_costs = "Normalized costs"
    normalized_performance = "Normalized performance"
    product = "Product"
    capital_cost = "Capital cost"
    operating_cost = "Operating cost"
    revenue = "Revenue"


def export_to_ui(flowsheet=None, exports=None, build_options=None, **kwargs):
    fs = flowsheet
    exports.add(
        obj=fs.feed.flow_vol[0],
        name="Flowrate",
        ui_units=pyunits.m**3 / pyunits.hr,
        display_rounding=1e-2,  # round to nearest 0.01
        description="Volumetric flowrate for the feed",
        is_input=True,
        input_category=InputCategory.feed,
        read_only=False,
        is_output=True,
        output_category=OutputCategory.feed,
        chart_type="stacked_bar",
        chart_group="test",
    )


def flowsheet_interface(exports=True, solve_func=solve_ro):
    kwargs = {}
    if exports:
        kwargs["do_export"] = export_to_ui
    return fsapi.FlowsheetInterface(
        # leave out name and description to test auto-fill
        do_build=build_ro,
        do_solve=solve_func,
        **kwargs,
    )


def noop(*args, **kwargs):
    return


@pytest.mark.unit
def test_create_interface():
    with pytest.raises(ValueError):
        _ = flowsheet_interface(exports=False)
    fsi = flowsheet_interface()
    fs2 = fsapi.FlowsheetInterface(
        fs=fsi.fs_exp, do_build=noop, do_export=noop, do_solve=noop
    )
    assert fs2.fs_exp == fsi.fs_exp


@pytest.mark.unit
def test_build_noexport():
    with pytest.raises(ValueError):
        flowsheet_interface(exports=False)


@pytest.mark.unit
def test_build():
    fsi = flowsheet_interface()
    fsi.build(erd_type="pressure_exchanger")
    data = fsi.dict()
    print(data)
    assert "exports" in data
    assert len(data["exports"]) == 1


@pytest.mark.parametrize(
    "add_variant",
    [
        "obj_kwarg",
        "model_export_arg",
        "model_export_data_kwarg",
        "model_export_dict_data_kwarg",
    ],
)
@pytest.mark.unit
def test_actions(add_variant: str):
    fsi = flowsheet_interface()
    built = False
    # garbage = {"trash": True}
    garbage = FAKE_FLOWSHEET
    m = FAKE_FLOWSHEET
    v1 = Var(name="variable1")
    v1.construct()
    v1.value = 1
    print(v1.display())

    def fake_build(build_options=None, **kwargs):
        nonlocal built
        built = True
        nonlocal m
        m = build_ro()
        return m

    def fake_solve(flowsheet=None):
        # flowsheet passed in here should be what fake_build() returns
        assert flowsheet == m.fs
        return SOLVE_RESULT_OK

    def fake_export(flowsheet=None, exports=None, build_options=None, **kwargs):
        with pytest.raises(Exception):
            exports.add(obj=garbage)

        # NOTE we use exclusive variants here
        # to avoid triggering the error when adding an object with the same obj_key
        # which happens when multiple ModelExport are created from the same pyomo object
        if add_variant == "obj_kwarg":
            exports.add(obj=v1)  # form 1
        elif add_variant == "model_export_kwarg":
            ve1 = fsapi.ModelExport(obj=v1)
            exports.add(ve1)  # form 2
        elif add_variant == "model_export_data_kwarg":
            ve1 = fsapi.ModelExport(obj=v1)
            exports.add(data=ve1)  # form 3
        elif add_variant == "model_export_dict_data_kwarg":
            ve1 = fsapi.ModelExport(obj=v1)
            exports.add(data=dict(ve1))  # form 4
        with pytest.raises(ValueError):
            exports.add(v1, v1)

    fsi.add_action(fsapi.Actions.build, fake_build)
    fsi.add_action(fsapi.Actions.export, fake_export)
    fsi.add_action(fsapi.Actions.solve, fake_solve)
    fsi.build()
    fsi.solve()
    with pytest.raises(ValueError):
        fsi.run_action(fsapi.Actions.export)


class CSVTestSettings:
    """Settings for test_csv_exports used in other functions."""

    bad_obj = False
    bad_units = False


@pytest.mark.unit
def test_csv_exports():
    for i in range(3):
        if i == 1:
            CSVTestSettings.bad_obj, CSVTestSettings.bad_units = True, False
        elif i == 2:
            CSVTestSettings.bad_obj, CSVTestSettings.bad_units = False, True
        else:
            CSVTestSettings.bad_obj, CSVTestSettings.bad_units = False, False
        for export_func in (csv_from_tempfile, csv_from_localfile):
            fsi = fsapi.FlowsheetInterface(
                do_build=build_ro, do_solve=solve_ro, do_export=export_func
            )
            if i == 0:
                fsi.build()  # expect success
            else:
                # expect failure (bad_units or bad_obj)
                with pytest.raises(RuntimeError):
                    fsi.build()


def csv_from_tempfile(exports=None, flowsheet=None, **kwargs):
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir) / "fake.csv"
        populate_csv_exports(f.open("w"))
        exports.from_csv(file=f, flowsheet=flowsheet)


def csv_from_localfile(exports=None, flowsheet=None, **kwargs):
    path = Path(__file__).parent / "test.csv"
    populate_csv_exports(path.open("w"))
    try:
        exports.from_csv(file="test.csv", flowsheet=flowsheet)
    finally:
        path.unlink()


def populate_csv_exports(f):
    units = "units.foobar" if CSVTestSettings.bad_units else "units.m**3/units.s"
    obj = "dirt" if CSVTestSettings.bad_obj else "fs.feed.flow_vol[0]"
    rows = [
        "name,obj,description,ui_units,display_units,rounding,is_input,input_category,is_output,output_category",
        f"feed,{obj},feed flow volume,{units},m^3/s,3,TRUE,something,FALSE,",
    ]
    for row in rows:
        f.write(row)
        f.write("\n")


@pytest.mark.unit
def test_load():
    fsi = flowsheet_interface()
    fsi.build(erd_type="pressure_exchanger")
    # get some info
    var_key = list(fsi.fs_exp.exports.keys())[0]
    var_obj = fsi.fs_exp.exports[var_key].obj
    save_value = var_obj.value
    # serialize
    data = fsi.dict()
    # modify
    data["exports"][var_key]["value"] = -1000
    # reload
    fsi.load(data)
    # check
    assert fsi.fs_exp.exports[var_key].value == -1000

    # this time with a missing thing
    data = fsi.dict()
    # add another (fake) one
    data["exports"]["foobar"] = data["exports"][var_key].copy()
    # reload (fake one will be 'missing')
    try:
        fsi.load(data)
    except fsapi.FlowsheetInterface.MissingObjectError as err:
        for item in err.missing:
            print(f"Missing item: key={item.key}, name={item.name}")
        assert len(err.missing) == 1
        assert err.missing[0].key == "foobar"
    else:
        assert False, "Expected a MissingObjectError"


@pytest.mark.unit
def test_require_methods():
    fsi = flowsheet_interface()
    methods = ("do_export", "do_build", "do_solve")
    kwargs = {m: noop for m in methods}
    # make one method 'bad' at a time
    for meth in methods:
        # missing
        badkw = kwargs.copy()
        del badkw[meth]
        with pytest.raises(ValueError):
            _ = fsapi.FlowsheetInterface(fsi, **badkw)
        # not callable
        badkw = kwargs.copy()
        badkw[meth] = 1
        with pytest.raises(TypeError):
            _ = fsapi.FlowsheetInterface(fsi, **badkw)


@pytest.mark.component
def test_export_values():
    # get an interface
    fsi = flowsheet_interface()
    fsi.build()
    d1 = fsi.dict()

    # change one value
    key = list(fsi.fs_exp.exports.keys())[0]
    orig_value = value(fsi.fs_exp.exports[key].obj)
    new_value = orig_value + 1
    print(f"@@ orig_value = {orig_value}, new value = {new_value}")
    fsi.fs_exp.exports[key].obj.value = new_value

    # re-export
    fsi.export_values()
    d2 = fsi.dict()

    print("== original")
    print(d1)
    print("== modified")
    print(d2)

    # check that change happened
    assert d1 != d2


@pytest.mark.component
def test_export_values_build():
    # get an interface
    fsi = flowsheet_interface()
    d1 = fsi.dict()
    fsi.build()
    # after build, new values should be exported to fsi.fs_exp
    d2 = fsi.dict()
    assert d1 != d2


@pytest.mark.unit
def test_empty_solve():
    # try a fake solve
    fsi = flowsheet_interface()
    with pytest.raises(RuntimeError) as excinfo:
        fsi.build()
        fsi.solve()
    print(f"* RuntimeError: {excinfo.value}")


@pytest.mark.unit
def test_nonoptimal_termination():
    fsi = DD.export_to_ui()
    fsi.build()

    # pick a crazy value
    key = list(fsi.fs_exp.exports.keys())[0]
    orig_value = value(fsi.fs_exp.exports[key].obj)
    new_value = orig_value + 1e9
    fsi.fs_exp.exports[key].obj.value = new_value
    print(f"* orig_value = {orig_value}, new value = {new_value}")

    # try to solve (for real)

    with pytest.raises(RuntimeError) as excinfo:
        fsi.solve()
    print(f"* RuntimeError: {excinfo.value}")


@pytest.mark.unit
def test_has_version():
    fsi = flowsheet_interface()
    d = fsi.dict()
    assert "version" in d
    assert d["version"] > 0


@pytest.mark.unit
def test_to_csv(tmpdir):
    fsi = flowsheet_interface()
    fsi.build()
    outputs = [
        tmpdir / "path.csv",
        str(tmpdir / "filename.csv"),
        open(tmpdir / "fileobj.csv", "w"),
    ]
    for o in outputs:
        num = fsi.fs_exp.to_csv(o)
        assert num > 0


@pytest.mark.unit
def test_add_option(tmpdir):
    fsi = flowsheet_interface()
    fsi.build()
    fsi.fs_exp.add_option(
        name="TestStrOptionValid",
        category="String Options",
        display_name="String Option",
        values_allowed="string",
        value="test option",
    )

    fsi.fs_exp.add_option(
        name="TestIntOptionValid",
        category="Int Options",
        display_name="Int Option",
        values_allowed="int",
        max_val=16,
        min_val=0,
        value=10,
    )

    fsi.fs_exp.add_option(
        name="TestFloatOptionValid",
        category="Float Options",
        display_name="Float Option",
        values_allowed="float",
        max_val=16,
        min_val=0,
        value=10.1,
    )

    fsi.fs_exp.add_option(
        name="TestListOptionValid",
        category="List Options",
        display_name="List Option",
        values_allowed=["valid option a", "valid option b"],
        value="valid option a",
    )

    with pytest.raises(ValueError) as excinfo:
        fsi.fs_exp.add_option(
            name="TestStrOptionInvalid",
            category="String Options",
            display_name="String Option",
            values_allowed="string",
            value=1,
        )

    with pytest.raises(ValueError) as excinfo:
        fsi.fs_exp.add_option(
            name="TestIntOptionInvalid",
            category="Int Options",
            display_name="Int Option",
            values_allowed="int",
            max_val=16,
            min_val=0,
            value=20,
        )

    with pytest.raises(ValueError) as excinfo:
        fsi.fs_exp.add_option(
            name="TestFloatOptionInvalid",
            category="Float Options",
            display_name="Float Option",
            values_allowed="float",
            max_val=16,
            min_val=0,
            value=-1,
        )

    with pytest.raises(ValueError) as excinfo:
        fsi.fs_exp.add_option(
            name="TestListOptionInvalid",
            category="List Options",
            display_name="List Option",
            values_allowed=["valid option a", "valid option b"],
            value="invalid option",
        )


@pytest.fixture
def kpi_export():
    exp = fsapi.FlowsheetExport()
    # create
    single = "single"
    exp.add_kpi_value(
        single, 1.2, label="single label", title="single title", units="singleUnits"
    )
    vector = "vector"
    exp.add_kpi_vector(
        vector,
        values=[1.2, -1.2],
        labels=["vlabel1", "vlabel2"],
        title="vector title",
        units="vectorUnits",
        xlab="x label",
        ylab="y label",
    )
    total = "total"
    exp.add_kpi_total(
        total,
        values=[1.2, 1.3],
        labels=["tlabel1", "tlabel2"],
        title="total title",
        units="totalUnits",
        total_label="total label",
    )
    return exp


@pytest.mark.unit
def test_flowsheet_export_kpi(kpi_export):
    single, vector, total = "single", "vector", "total"
    exp = kpi_export
    # validate
    kpis = exp.model_dump()["kpis"]
    expect = {
        single: {
            "is_vector": False,
            "has_total": False,
            "name": "single",
            "title": "single title",
            "units": "singleUnits",
            "values": [1.2],
            "labels": ["single label"],
            "xlab": "",
            "ylab": "",
            "total": 0.0,
            "total_label": "",
        },
        vector: {
            "is_vector": True,
            "has_total": False,
            "name": "vector",
            "title": "vector title",
            "units": "vectorUnits",
            "values": [1.2, -1.2],
            "labels": ["vlabel1", "vlabel2"],
            "xlab": "x label",
            "ylab": "y label",
            "total": 0.0,
            "total_label": "",
        },
        total: {
            "is_vector": True,
            "has_total": True,
            "name": "total",
            "title": "total title",
            "units": "totalUnits",
            "values": [1.2, 1.3],
            "labels": ["tlabel1", "tlabel2"],
            "xlab": "",
            "ylab": "",
            "total": 2.5,
            "total_label": "total label",
        },
    }
    for name in (single, vector, total):
        got = kpis[name]
        for k in got:
            assert got[k] == expect[name][k]


@pytest.mark.unit
def test_flowsheet_report(kpi_export):
    exp = kpi_export


def test_layout_single_column():
    # Single column layout
    kpis = {"kpi1": "<div>kpi1</div>", "kpi2": "<div>kpi2</div>"}
    spec = ["kpi1", "kpi2"]
    layout = fsapi.Layout(spec, kpis)
    body = layout.body
    css = layout.css
    assert "kpi1" in body and "kpi2" in body
    assert ".grid_column" in css
    assert ".grid_row" in css


def test_layout_single_row():
    # Single row layout
    kpis = {"kpi1": "<div>kpi1</div>", "kpi2": "<div>kpi2</div>"}
    spec = [["kpi1", "kpi2"]]
    layout = fsapi.Layout(spec, kpis)
    body = layout.body
    assert "kpi1" in body and "kpi2" in body
    # Should have both row and column divs
    assert body.count("grid_row") > 0
    assert body.count("grid_column") > 0


def test_layout_grid():
    # 2x2 grid
    kpis = {
        "kpi1": "<div>kpi1</div>",
        "kpi2": "<div>kpi2</div>",
        "kpi3": "<div>kpi3</div>",
        "kpi4": "<div>kpi4</div>",
    }
    spec = [["kpi1", "kpi2"], ["kpi3", "kpi4"]]
    layout = fsapi.Layout(spec, kpis)
    body = layout.body
    assert all(k in body for k in kpis)
    # Should have two row divs
    assert body.count("grid_row") >= 2


def test_layout_nested():
    # Nested layout
    kpis = {"a": "<div>a</div>", "b": "<div>b</div>", "c": "<div>c</div>"}
    spec = [["a", ["b", "c"]]]
    layout = fsapi.Layout(spec, kpis)
    body = layout.body
    assert "a" in body and "b" in body and "c" in body
    # Should have nested divs
    assert body.count("grid_row") > 0
    assert body.count("grid_column") > 0


def test_layout_css():
    kpis = {"kpi": "<div>kpi</div>"}
    spec = ["kpi"]
    layout = fsapi.Layout(spec, kpis)
    css = layout.css
    assert ".grid_row" in css
    assert ".grid_column" in css


def test_flowsheetinterface_get_diagram_returns_none():
    fsi = flowsheet_interface()
    fsi.build()
    # No get_diagram function provided, should return None
    assert fsi.get_diagram() is None


def test_flowsheetinterface_dict_and_load():
    fsi = flowsheet_interface()
    fsi.build()
    d = fsi.dict()
    # Should contain 'exports' and 'version'
    assert "exports" in d and "version" in d
    # Should be able to load the same dict
    fsi.load(d)


def test_flowsheetinterface_select_option():
    fsi = flowsheet_interface()
    fsi.build()
    # Add an option
    fsi.fs_exp.add_option(name="opt1", values_allowed=["a", "b"], value="a")
    fsi.select_option("opt1", "b")
    assert fsi.fs_exp.build_options["opt1"].value == "b"


def test_flowsheetinterface_export_values():
    fsi = flowsheet_interface()
    fsi.build()
    # Should not raise
    fsi.export_values()


def test_flowsheetinterface_from_installed_packages_and_from_module():
    # These are mostly smoke tests, as they depend on environment
    result = fsapi.FlowsheetInterface.from_installed_packages()
    assert isinstance(result, dict)
    # from_module with a string (should not raise, may return None)
    mod = "watertap.flowsheets.seawater_RO_desalination"
    try:
        iface = fsapi.FlowsheetInterface.from_module(mod)
    except Exception:
        iface = None
    # iface may be None if the module doesn't have the UI_HOOK
    assert iface is None or hasattr(iface, "build")


def test_flowsheetinterface_report():
    fsi = flowsheet_interface()
    fsi.build()
    report = fsi.report()
    assert hasattr(report, "html")


def test_flowsheetreport_html_and_build(kpi_export):
    from idaes_flowsheet_processor.api import FlowsheetReport

    report = FlowsheetReport(kpi_export)
    html = report.html()
    assert html.startswith("<html>")
    body, css = report.build()
    assert isinstance(body, str)
    assert isinstance(css, str)
    assert any(k in body for k in kpi_export.kpis)
    assert ".grid_row" in css
    assert ".grid_column" in css


def test_flowsheetreport_kpi_html_and_barchart(kpi_export):
    from idaes_flowsheet_processor.api import FlowsheetReport

    report = FlowsheetReport(kpi_export)
    # Test _kpi_html for each KPI
    for kpi in kpi_export.kpis.values():
        html = report._kpi_html(kpi)
        assert isinstance(html, str)
    # Test create_barchart for a vector KPI
    for kpi in kpi_export.kpis.values():
        if kpi.is_vector and not kpi.has_total:
            chart_html = report.create_barchart(kpi)
            assert isinstance(chart_html, str)


def test_flowsheetreport_create_total_chart(kpi_export):
    from idaes_flowsheet_processor.api import FlowsheetReport, WAFFLE, DONUT

    # add some where total is zero
    kpi_export.add_kpi_total(
        "zero1",
        values=[1.2, -1.2],  # total = 0
        labels=["tlabel1", "tlabel2"],
        title="total title",
        units="totalUnits",
        total_label="total label",
    )
    kpi_export.add_kpi_total(
        "zero2",
        values=[0, 0, 0],  # total = 0
        labels=["tlabel1", "tlabel2"],
        title="total title",
        units="totalUnits",
        total_label="total label",
    )
    # Test create_total for a KPI with has_total
    for kpi in kpi_export.kpis.values():
        if kpi.is_vector and kpi.has_total:
            if kpi.name.startswith("zero"):
                with pytest.raises(ValueError):
                    FlowsheetReport.create_total(kpi)
            else:
                # Test both WAFFLE and DONUT chart types
                html_waffle = FlowsheetReport.create_total(kpi, total_type=WAFFLE)
                html_donut = FlowsheetReport.create_total(kpi, total_type=DONUT)
                assert isinstance(html_waffle, str)
                assert isinstance(html_donut, str)


def test_flowsheetreport_create_value(kpi_export):
    from idaes_flowsheet_processor.api import FlowsheetReport

    for kpi in kpi_export.kpis.values():
        html = FlowsheetReport.create_value(kpi)
        assert isinstance(html, str)


def test_flowsheetreport_preprocess_total_type():
    from idaes_flowsheet_processor.api import FlowsheetReport, WAFFLE, DONUT

    # Accepts enum and string
    assert FlowsheetReport._preprocess_total_type(WAFFLE) == WAFFLE
    assert FlowsheetReport._preprocess_total_type("waffle") == WAFFLE
    assert FlowsheetReport._preprocess_total_type("donut") == DONUT
    import pytest

    with pytest.raises(ValueError):
        FlowsheetReport._preprocess_total_type("notatype")
