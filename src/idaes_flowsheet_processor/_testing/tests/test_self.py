import pytest


pytest.importorskip(
    "watertap.flowsheets",
    reason="testing of the pytest plugin is currently not possible without WaterTAP flowsheets",
)

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    "cli_args",
    [
        pytest.param(
            ["--idaes-flowsheets", "--entry-points-group", "watertap.flowsheets"],
            id="with entry points",
        ),
        pytest.param(
            [
                "--idaes-flowsheets",
                "--entry-points-group",
                "watertap.flowsheets",
                "--modules",
                "watertap.flowsheet.mvc.mvc_single_stage_ui",
            ],
            id="with entry points and a single module",
        ),
        pytest.param(
            [
                "--idaes-flowsheets",
                "--modules",
                "watertap.flowsheet.mvc.mvc_single_stage_ui",
            ],
            id="with a single module",
        ),
        pytest.param(
            [
                "--idaes-flowsheets",
                "--modules",
                "watertap.flowsheet.mvc.mvc_single_stage_ui",
                "watertap.flowsheets.gac.gac_ui",
            ],
            id="with multiple modules",
        ),
    ],
)
def test_plugin_collects_successfully(cli_args: list[str], pytester: pytest.Pytester):
    cli_args.append("--collect-only")
    res = pytester.runpytest_subprocess(*cli_args)
    assert res.ret == 0
