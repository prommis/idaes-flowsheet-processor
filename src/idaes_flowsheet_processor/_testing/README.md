# `idaes-flowsheets` pytest plugin

## Setup

The plugin will be automatically installed alongside the `idaes-flowsheet-processor` Python package.

## Usage

The plugin is only active in a pytest run when the `--idaes-flowsheet` CLI flag is provided.

Here are some usage examples:

```sh
# this will just run pytest WITHOUT the `idaes-flowsheets` plugin
pytest

# the plugin is enabled, but no flowsheet sources are given, so it won't do much
pytest --idaes-flowsheets

# with `--entry-points-group`, all flowsheet interfaces defined under the given entry point group will be collected and tested
pytest --idaes-flowsheets --entry-points-group watertap.flowsheets

# with `--modules`, any importable Python modules (that might or might not be associated with an entry point) will be loaded
pytest --idaes-flowsheets --modules my_package.my_module

# multiple modules can also be specified
pytest --idaes-flowsheets --modules my_package.my_first_module my_other_package.my_second_module

# providing both `--entry-points-group` and `modules` will work; however, note that no attempt at deduplicating will be done (i.e. if a module is both provided directly and also part of the entry point group, it will be collected and tested twice)
pytest --idaes-flowsheets --entry-points-group watertap.flowsheet --modules my_package.my_first_module my_other_package.my_second_module

# it is also possible to use a custom test class to test each collected flowsheet interface, provided as the `flowsheet_interface` class-scoped fixture
pytest --idaes-flowsheets --entry-points-group watertap.flowsheet --modules my_package.my_first_module my_other_package.my_second_module --test-class my_package.my_flowsheet_testing_utils:TestFlowsheetInterface
```

For more detailed information, run `pytest --help` and search for the `idaes-flowsheets` paragraph.
