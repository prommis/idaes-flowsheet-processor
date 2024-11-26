import pytest
from idaes_flowsheet_processor.api import FlowsheetInterface


class TestFlowsheetInterface:

    def test_instance(self, flowsheet_interface):
        assert isinstance(flowsheet_interface, FlowsheetInterface)

    @pytest.fixture(scope="class")
    def interface_post_build(self, flowsheet_interface):
        flowsheet_interface.build()
        return flowsheet_interface

    @pytest.fixture(scope="class")
    def data(self, interface_post_build):
        return interface_post_build.dict()

    def test_data(self, data):
        assert len(data) > 0

    @pytest.fixture(scope="class")
    def exports(self, data):
        return data.get("exports", None)

    def test_exports(self, exports):
        assert exports is not None
        assert len(exports) >= 1

    @pytest.mark.solver
    def test_solve(self, interface_post_build):
        interface_post_build.solve()
