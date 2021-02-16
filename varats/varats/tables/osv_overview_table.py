"""Module for writing bug-data metrics tables."""
import typing as tp

import numpy as np
import pandas as pd
from tabulate import tabulate

from varats.data.databases.osv_database import OSVDatabase
from varats.project.project_util import get_project_cls_by_name
from varats.provider.bug.bug_provider import BugProvider
from varats.provider.osv.osv_provider import OSVProvider
from varats.table.table import Table, TableFormat, wrap_table_in_document


class BugOverviewTable(Table):
    """Visualizes osv metrics of a project."""

    NAME = "osv_overview"

    def __init__(self, **kwargs: tp.Any):
        super().__init__(self.NAME, **kwargs)

    def tabulate(self) -> str:
        project_name = self.table_kwargs["project"]

        variables = ["osv_id", "severity", "affected_ranges"]
        vulnerabilities = OSVDatabase.get_data_for_project(
            project_name, variables
        )

        if self.format in [
            TableFormat.latex, TableFormat.latex_raw, TableFormat.latex_booktabs
        ]:
            tex_code = vulnerabilities.to_latex(
                bold_rows=True, multicolumn_format="c", longtable=True
            )
            return str(tex_code) if tex_code else ""
        return tabulate(
            vulnerabilities, vulnerabilities.columns, self.format.value
        )

    def wrap_table(self, table: str) -> str:
        return wrap_table_in_document(table=table, landscape=True)
