"""Module for diff based commit-data metrics."""
import typing as tp
from datetime import datetime
from itertools import chain
from pathlib import Path

import pandas as pd
from benchbuild.utils.revision_ranges import (
    AbstractRevisionRange,
    SingleRevision,
    RevisionRange,
    GoodBadSubgraph,
)

from varats.data.cache_helper import (
    build_cached_report_table,
    get_data_file_path,
)
from varats.data.databases.evaluationdatabase import EvaluationDatabase
from varats.data.reports.blame_report import (
    BlameReport,
    BlameReportDiff,
    generate_degree_tuples,
    generate_author_degree_tuples,
    generate_avg_time_distribution_tuples,
    generate_max_time_distribution_tuples,
    count_interactions,
    count_interacting_commits,
    count_interacting_authors,
)
from varats.jupyterhelper.file import load_blame_report
from varats.mapping.commit_map import CommitMap
from varats.paper.case_study import CaseStudy
from varats.paper_mgmt.case_study import get_case_study_file_name_filter
from varats.project.project_util import (
    get_local_project_git,
    get_project_cls_by_name,
)
from varats.provider.osv.osv import OSVVulnerability, OSVRange
from varats.provider.osv.osv_provider import OSVProvider
from varats.report.report import MetaReport
from varats.revision.revisions import (
    get_processed_revisions_files,
    get_failed_revisions_files,
    get_processed_revisions,
)
from varats.utils.git_util import (
    ChurnConfig,
    calc_code_churn,
    create_commit_lookup_helper,
)


class OSVDatabase():
    """Database for OSV vulnerabilities."""

    CACHE_ID = "osv_vulnerabilities"
    COLUMNS = ["osv_id", "severity", "affected_ranges"]

    @staticmethod
    def _vulnerability_id(vulnerability: OSVVulnerability) -> str:
        return vulnerability.id

    @staticmethod
    def _vulnerability_timestamp(vulnerability: OSVVulnerability) -> str:
        # not needed since we assume vulnerabilities to be immutable
        return ""

    @staticmethod
    def _compare_timestamps(ts1: str, ts2: str) -> bool:
        # always false since we assume vulnerabilities to be immutable
        return False

    @staticmethod
    def ranges_to_str(ranges: tp.Set[OSVRange]) -> str:
        return ";".join([
            f"{str(range.introduced_in)}-{str(range.fixed_in)}"
            for range in ranges
        ])

    @staticmethod
    def __parse_revision_range(range_str: str) -> AbstractRevisionRange:
        if ":" in range_str:
            return RevisionRange(*range_str.split(":", maxsplit=1))
        if "\\" in range_str:
            bad, good = range_str.split("\\", maxsplit=1)
            return GoodBadSubgraph(bad.split(","), good.split(","))
        return SingleRevision(range_str)

    @staticmethod
    def ranges_from_str(ranges_str: str) -> tp.Set[OSVRange]:
        introducer_fix_tuples = [
            ranges_str.split("-", maxsplit=1)
            for range_str in ranges_str.split(";")
        ]
        ranges: tp.Set[OSVRange] = set()
        for tuple in introducer_fix_tuples:
            introducer = tuple[0]
            fix = tuple[1]
            ranges.add(
                OSVRange(
                    OSVDatabase.__parse_revision_range(introducer),
                    OSVDatabase.__parse_revision_range(fix)
                )
            )

        return ranges

    @classmethod
    def _load_dataframe(
        cls, project_name: str, **kwargs: tp.Any
    ) -> pd.DataFrame:

        def create_dataframe_layout() -> pd.DataFrame:
            df_layout = pd.DataFrame(columns=cls.COLUMNS)
            return df_layout

        def create_dataframe_from_vulnerability(
            vulnerability: OSVVulnerability
        ) -> tp.Tuple[pd.DataFrame, str, str]:
            ranges_str = OSVDatabase.ranges_to_str(
                vulnerability.affected_ranges
            )

            return (
                pd.DataFrame({
                    "osv_id": vulnerability.id,
                    "severity": vulnerability.severity,
                    "affected_ranges": ranges_str
                }), OSVDatabase._vulnerability_id(vulnerability),
                OSVDatabase._vulnerability_timestamp(vulnerability)
            )

        vulnerabilities = OSVProvider.get_provider_for_project(
            get_project_cls_by_name(project_name)
        ).get_all_vulnerabilities()

        # cls.CACHE_ID is set by superclass
        # pylint: disable=E1101
        data_frame = build_cached_report_table(
            cls.CACHE_ID, project_name, vulnerabilities, [],
            create_dataframe_layout, create_dataframe_from_vulnerability,
            OSVDatabase._vulnerability_id, OSVDatabase._vulnerability_timestamp,
            OSVDatabase._compare_timestamps
        )

        return data_frame

    @classmethod
    def get_data_for_project(
        cls, project_name: str, columns: tp.List[str], **kwargs: tp.Any
    ) -> pd.DataFrame:
        data: pd.DataFrame = cls._load_dataframe(project_name, **kwargs)

        if not [*data] == cls.COLUMNS:
            raise AssertionError(
                "Loaded dataframe does not match expected layout."
                "Consider removing the cache file "
                f"{get_data_file_path(cls.CACHE_ID, project_name)}."
            )

        if not all(column in cls.COLUMNS for column in columns):
            raise ValueError(
                f"All values in 'columns' must be in {cls.__name__}.COLUMNS"
            )

        return data[columns]
