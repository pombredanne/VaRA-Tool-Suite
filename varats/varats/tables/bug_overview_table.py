"""Module for writing bug-data metrics tables."""
import statistics
import typing as tp
from datetime import date, datetime

import numpy as np
import pandas as pd
import pygit2
from scipy import stats
from tabulate import tabulate

from varats.data.reports.szz_report import SZZUnleashedReport
from varats.project.project_util import (
    get_project_cls_by_name,
    get_local_project_git,
)
from varats.provider.bug.bug import RawBug, PygitBug
from varats.provider.bug.bug_provider import BugProvider
from varats.revision.revisions import get_processed_revisions_files
from varats.table.table import Table, TableFormat, wrap_table_in_document


class BugOverviewTable(Table):
    """Visualizes bug metrics of a project."""

    NAME = "bug_overview"

    def __init__(self, **kwargs: tp.Any):
        super().__init__(self.NAME, **kwargs)

    def tabulate(self) -> str:
        project_name = self.table_kwargs["project"]

        bug_provider = BugProvider.get_provider_for_project(
            get_project_cls_by_name(project_name)
        )

        variables = [
            "fixing hash", "fixing message", "fixing author", "issue_number"
        ]
        pybugs = bug_provider.find_all_pygit_bugs()

        data_rows = [[
            pybug.fixing_commit.hex, pybug.fixing_commit.message,
            pybug.fixing_commit.author.name, pybug.issue_id
        ] for pybug in pybugs]

        bug_df = pd.DataFrame(columns=variables, data=np.array(data_rows))

        if self.format in [
            TableFormat.latex, TableFormat.latex_raw, TableFormat.latex_booktabs
        ]:
            tex_code = bug_df.to_latex(
                bold_rows=True, multicolumn_format="c", longtable=True
            )
            return str(tex_code) if tex_code else ""
        return tabulate(bug_df, bug_df.columns, self.format.value)

    def wrap_table(self, table: str) -> str:
        return wrap_table_in_document(table=table, landscape=True)


class BugFixingEvaluationTable(Table):
    """
    Visualizes true positives and false positives for a file of given fixing
    commits.

    Provide a file with true fixing commits as an extra argument `fixes_path`,
    else the list of true fixing commits will be considered none. Set extra
    argument `starting_from` to set a date from when to consider bug fixes.
    """

    NAME = "bug_fixing_evaluation"

    def __init__(self, **kwargs: tp.Any):
        super().__init__(self.NAME, **kwargs)

    def tabulate(self) -> str:
        project_name = self.table_kwargs["project"]
        fixes_file_name = self.table_kwargs.get("fixes_path", "")
        #Format for starting_from parameter: month/day/year
        start_date = datetime.strptime(
            self.table_kwargs["starting_from"], '%m/%d/%Y'
        ) if "starting_from" in self.table_kwargs else date.today()

        input_fixing_commits: tp.Set[str] = set()
        with open(fixes_file_name) as input_file:
            input_fixing_commits: tp.Set[str] = set(
                line.rstrip() for line in input_file
            )

        bug_provider = BugProvider.get_provider_for_project(
            get_project_cls_by_name(project_name)
        )

        variables = [
            "commits total", "true fixes", "fixes found", "true positive",
            "false positive", "true negative", "false negative"
        ]
        annotations = ["fix(e[ds])?"]
        rawbugs = bug_provider.find_all_raw_bugs()

        fixing_eval = _evaluate_fixing_commits(
            project_name, start_date, rawbugs, input_fixing_commits
        )

        data: tp.List[int] = [[
            len(fixing_eval[i]) for i in range(len(variables))
        ]]

        eval_df = pd.DataFrame(data=data, columns=variables, index=annotations)

        if self.format in [
            TableFormat.latex, TableFormat.latex_raw, TableFormat.latex_booktabs
        ]:
            tex_code = eval_df.to_latex(bold_rows=True, multicolumn_format="c")
            return str(tex_code) if tex_code else ""
        return tabulate(eval_df, eval_df.columns, self.format.value)

    def wrap_table(self, table: str) -> str:
        return wrap_table_in_document(table=table, landscape=True)


class BugIntroducingEvaluationTable(Table):
    """
    Visualizes different metrics on introducing commits of bugs without ground
    truth.

    Set extra argument `starting_from` to set a date from when to consider bug
    fixes.
    """

    NAME = "bug_introducing_evaluation"

    def __init__(self, **kwargs: tp.Any):
        super().__init__(self.NAME, **kwargs)

    def tabulate(self) -> str:
        project_name = self.table_kwargs["project"]

        start_date = datetime.strptime(
            self.table_kwargs["starting_from"], '%m/%d/%Y'
        ) if "starting_from" in self.table_kwargs else date.today()

        bug_provider = BugProvider.get_provider_for_project(
            get_project_cls_by_name(project_name)
        )
        rawbugs = bug_provider.find_all_raw_bugs()

        variables = [
            "realism intro", "realism intro %", "futimp time span",
            "futimp time span %", "futimp count", "futimp count %"
        ]
        annotations = ["fix(e[ds])?"]

        rawbugs_filtered = _get_bugs_fixed_after_threshold(
            project_name, rawbugs, start_date
        )

        med_tuple_realism = _compute_realism_of_introduction(
            project_name, rawbugs_filtered
        )
        med_tuple_impact_time_span = _compute_future_impact_time_span(
            project_name, rawbugs_filtered
        )
        med_tuple_impact_count = _compute_future_impact_count()

        passed_realism = _get_passing_fraction_realism_of_introduction(
            project_name, rawbugs_filtered, med_tuple_realism[0],
            med_tuple_realism[1]
        )
        passed_impact_time_span = _get_passing_fraction_future_impact_time_span(
            project_name, rawbugs_filtered, med_tuple_impact_time_span[0],
            med_tuple_impact_time_span[1]
        )
        passed_impact_count = _get_passing_fraction_future_impact_count(
            project_name, rawbugs_filtered, med_tuple_impact_count[0],
            med_tuple_impact_count[1]
        )

        data = [[
            med_tuple_realism[0], passed_realism, med_tuple_impact_time_span[0],
            passed_impact_time_span, med_tuple_impact_count[0],
            passed_impact_count
        ]]

        eval_df = pd.DataFrame(data=data, columns=variables, index=annotations)

        if self.format in [
            TableFormat.latex, TableFormat.latex_raw, TableFormat.latex_booktabs
        ]:
            tex_code = eval_df.to_latex(bold_rows=True, multicolumn_format="c")
            return str(tex_code) if tex_code else ""
        return tabulate(eval_df, eval_df.columns, self.format.value)

    def wrap_table(self, table: str) -> str:
        return wrap_table_in_document(table=table, landscape=True)


class SZZComparisonTable(Table):
    """Table that compares the bug data of the szz approaches SZZUnleashed and
    BugProvider."""

    NAME = "szz_comparison"

    def __init__(self, **kwargs: tp.Any):
        super().__init__(self.NAME, **kwargs)

    def tabulate(self) -> str:
        project_name = self.table_kwargs["project"]

        bug_provider = BugProvider.get_provider_for_project(
            get_project_cls_by_name(project_name)
        )

        provider_rawbugs = bug_provider.find_all_raw_bugs()

        reports = get_processed_revisions_files(
            project_name, SZZUnleashedReport
        )
        szzunleashed_rawbugs = SZZUnleashedReport(reports[0]).get_all_raw_bugs()

        med_provider_realism = _compute_realism_of_introduction(
            project_name, provider_rawbugs
        )
        med_unleashed_realism = _compute_realism_of_introduction(
            project_name, szzunleashed_rawbugs
        )

        med_provider_impact_time_span = _compute_future_impact_time_span(
            project_name, provider_rawbugs
        )
        med_unleashed_impact_time_span = _compute_future_impact_time_span(
            project_name, szzunleashed_rawbugs
        )

        med_provider_impact_count = _compute_future_impact_count()
        med_unleashed_impact_count = _compute_future_impact_count()

        passed_provider_realism = _get_passing_fraction_realism_of_introduction(
            project_name, provider_rawbugs, med_provider_realism[0],
            med_provider_realism[1]
        )
        passed_unleashed_realism = _get_passing_fraction_realism_of_introduction(
            project_name, szzunleashed_rawbugs, med_unleashed_realism[0],
            med_unleashed_realism[1]
        )

        passed_provider_impact_time_span = _get_passing_fraction_future_impact_time_span(
            project_name, provider_rawbugs, med_provider_impact_time_span[0],
            med_provider_impact_time_span[1]
        )
        passed_unleashed_impact_time_span = _get_passing_fraction_future_impact_time_span(
            project_name, szzunleashed_rawbugs,
            med_unleashed_impact_time_span[0], med_unleashed_impact_time_span[1]
        )

        passed_provider_impact_count = _get_passing_fraction_future_impact_count(
            project_name, provider_rawbugs, med_provider_impact_count[0],
            med_provider_impact_count[1]
        )
        passed_unleashed_impact_count = _get_passing_fraction_future_impact_count(
            project_name, szzunleashed_rawbugs, med_unleashed_impact_count[0],
            med_unleashed_impact_count[1]
        )

        variables = [
            "realism intro", "realism intro %", "futimp time span",
            "futimp time span %", "futimp count", "futimp count %"
        ]
        annotations = ["bugprovider", "szzunleashed"]

        data = [[
            med_provider_realism[0], passed_provider_realism,
            med_provider_impact_time_span[0], passed_provider_impact_time_span,
            med_provider_impact_count[0], passed_provider_impact_count
        ],
                [
                    med_unleashed_realism[0], passed_unleashed_realism,
                    med_unleashed_impact_time_span[0],
                    passed_unleashed_impact_time_span,
                    med_unleashed_impact_count[0], passed_unleashed_impact_count
                ]]

        eval_df = pd.DataFrame(data=data, columns=variables, index=annotations)

        if self.format in [
            TableFormat.latex, TableFormat.latex_raw, TableFormat.latex_booktabs
        ]:
            tex_code = eval_df.to_latex(bold_rows=True, multicolumn_format="c")
            return str(tex_code) if tex_code else ""
        return tabulate(eval_df, eval_df.columns, self.format.value)

    def wrap_table(self, table: str) -> str:
        return wrap_table_in_document(table=table, landscape=True)


def _evaluate_fixing_commits(
    project_name: str, start_date: datetime, rawbugs: tp.FrozenSet[RawBug],
    input_fixing_commits: tp.Set[str]
) -> tp.Tuple[tp.Set[str], tp.Set[str], tp.Set[str], tp.Set[str], tp.Set[str],
              tp.Set[str], tp.Set[str]]:
    """
    Output format:

    ( Set of commits in timespan, Set of true fixes in timespan, Set of found
    fixes in timespan, Set of true positives, Set of false positives, Set of
    true negatives, Set of false negatives )
    """
    project_repo = get_local_project_git(project_name)

    total_commits: tp.Set[str] = set()

    # sets of commit hashes for each category
    found_fixing_commits: tp.Set[str] = set()
    for rawbug in rawbugs:
        rawbug_pygit_fix = project_repo.revparse_single(rawbug.fixing_commit)
        if datetime.fromtimestamp(rawbug_pygit_fix.commit_time) >= start_date:
            found_fixing_commits.add(rawbug.fixing_commit)

    true_fixing_commits: tp.Set[str] = set()
    for commit_hash in input_fixing_commits:
        input_pygit_fix = project_repo.revparse_single(commit_hash)
        if datetime.fromtimestamp(input_pygit_fix.commit_time) >= start_date:
            true_fixing_commits.add(commit_hash)

    # also count total commits here
    found_non_fixing_commits: tp.Set[str] = set()
    for commit in project_repo.walk(
        project_repo.head.target.hex, pygit2.GIT_SORT_TIME
    ):
        if datetime.fromtimestamp(commit.commit_time) < start_date:
            break
        if commit.hex not in found_fixing_commits:
            found_non_fixing_commits.add(commit.hex)
        total_commits.add(commit.hex)

    # count bugs that are correctly labelled as fixing
    tp_commits = set()
    for commit in found_fixing_commits:
        if commit in true_fixing_commits:
            tp_commits.add(commit)

    # count bugs that are incorrectly labelled as fixing
    fp_commits = set()
    for commit in found_fixing_commits:
        if commit not in true_fixing_commits:
            fp_commits.add(commit)

    # count bugs that are correctly not labelled as fixing
    tn_commits = set()
    for commit in found_non_fixing_commits:
        if commit not in true_fixing_commits:
            tn_commits.add(commit)

    # count bugs that are incorrectly not labelled as fixing
    fn_commits = set()
    for commit in found_non_fixing_commits:
        if commit in true_fixing_commits:
            fn_commits.add(commit)

    return (
        total_commits, true_fixing_commits, found_fixing_commits, tp_commits,
        fp_commits, tn_commits, fn_commits
    )


def _get_passing_fraction_future_impact_count(
    project_name: str, rawbugs: tp.FrozenSet[RawBug], median: float, mad: float
) -> float:
    """Returns the fraction of how many introducing commits pass the future
    impact threshold (count of future bugs)."""
    intro_dict = _get_intro_dict(project_name, rawbugs)

    passing_intros: int = 0
    for introducer, fixing_commits in intro_dict.items():
        if len(fixing_commits) <= median + mad:
            passing_intros = passing_intros + 1

    return float(passing_intros) / float(len(intro_dict.keys()))


def _get_passing_fraction_future_impact_time_span(
    project_name: str, rawbugs: tp.FrozenSet[RawBug], median: float, mad: float
) -> float:
    """Returns the fraction of how many introducing commits pass the future
    impact threshold (time span of future bugs)."""
    intro_dict = _get_intro_dict(project_name, rawbugs)
    project_repo = get_local_project_git(project_name)

    passing_intros = 0
    for introducer, fixing_commits in intro_dict.items():
        introducer_pycommit: pygit2.Commit = project_repo.revparse_single(
            introducer
        )

        passed = True
        for fix in fixing_commits:
            fixing_pycommit: pygit2.Commit = project_repo.revparse_single(fix)

            introducer_date = datetime.fromtimestamp(
                introducer_pycommit.commit_time
            )
            fixing_date = datetime.fromtimestamp(fixing_pycommit.commit_time)

            if (abs(fixing_date - introducer_date).days > median + mad):
                passed = False
                break

        if passed:
            passing_intros = passing_intros + 1

    return float(passing_intros) / float(len(intro_dict.keys()))


def _get_passing_fraction_realism_of_introduction(
    project_name: str, rawbugs: tp.FrozenSet[RawBug], median: float, mad: float
) -> float:
    """Returns the fraction of how many fixing commits pass the realism of bug
    introduction threshold."""
    project_repo = get_local_project_git(project_name)

    passing_fixes = 0
    for rawbug in rawbugs:
        passed = True
        for intro_hash_a in rawbug.introducing_commits:
            introducer_a: pygit2.Commit = project_repo.revparse_single(
                intro_hash_a
            )
            intro_a_date = datetime.fromtimestamp(introducer_a.commit_time)
            for intro_hash_b in pybug.introducing_commits:
                introducer_b: pygit2.Commit = project_repo.revparse_single(
                    intro_hash_b
                )
                intro_b_date = datetime.fromtimestamp(introducer_b.commit_time)

                if (abs(intro_b_date - intro_a_date).days > median + mad):
                    passed = False

        if passed:
            passing_fixes = passing_fixes + 1

    return float(passing_fixes) / float(len(pybugs))


def _get_intro_dict(project_name: str,
                    pybugs: tp.FrozenSet[RawBug]) -> tp.Dict[str, tp.Set[str]]:
    intro_dict: tp.Dict[str, tp.Set[str]] = {}
    project_repo = get_local_project_git(project_name)

    for pybug in pybugs:
        fix_hash = pybug.fixing_commit
        for introducer in pybug.introducing_commits:
            if introducer not in intro_dict.keys():
                intro_dict[introducer] = set()
            intro_dict[introducer].add(fix_hash)

    return intro_dict


def _get_bugs_fixed_after_threshold(
    project_name: str, rawbugs: tp.FrozenSet[RawBug], start_date: datetime
) -> tp.FrozenSet[RawBug]:
    project_repo = get_local_project_git(project_name)
    rawbugs: tp.Set[RawBug] = set()

    for rawbug in rawbugs:
        pyfix: pygit2.Commit = project_repo.revparse_single(
            rawbug.fixing_commit
        )
        fixing_date = datetime.fromtimestamp(pyfix.commit_time)
        if fixing_date >= start_date:
            resulting_rawbugs.add(pybug)

    return frozenset(resulting_rawbugs)


def _count_commit_message_and_issue_event_bugs(
    rawbugs: tp.FrozenSet[RawBug]
) -> tp.Tuple[int, int]:
    """Counts the bugs found by commit messages and the bugs found by issue
    events."""
    message_bugs = 0
    issue_bugs = 0

    for rawbug in rawbugs:
        if rawbug.issue_id:
            issue_bugs = issue_bugs + 1
        else:
            message_bugs = message_bugs + 1

    return (message_bugs, issue_bugs)


def _compute_future_impact_count() -> tp.Tuple[float, float]:
    """
    Returns the median time difference and median absolute deviation between bug
    introductions and their fixes. Result determines the future impact of
    changes (count of future bugs). Note that this is statically set to (3,0)
    for all projects and exists for abstraction purposes.

    Args:
        rawbugs: The set of bugs to analyze

    Returns:
        A tuple (median, median absolute deviation)
    """
    return (3.0, 0.0)


def _compute_future_impact_time_span(
    project_name: str, rawbugs: tp.FrozenSet[RawBug]
) -> tp.Tuple[float, float]:
    """
    Computes the median time difference and median absolute deviation between
    bug introductions and their fixes. Result determines the future impact of
    changes (time span of future bugs).

    Args:
        rawbugs: The set of bugs to analyze

    Returns:
        A tuple (median, median absolute deviation)
    """
    project_repo = get_local_project_git(project_name)

    date_differences: tp.List[int] = []
    for rawbug in rawbugs:
        pyfix: pygit2.Commit = project_repo.revparse_single(
            rawbug.fixing_commit
        )
        fixing_date = datetime.fromtimestamp(pyfix.commit_time)
        for introducer in rawbug.introducing_commits:
            pyntro: pygit2.Commit = project_repo.revparse_single(introducer)
            intro_date = datetime.fromtimestamp(pyntro.commit_time)
            date_differences.append((fixing_date - intro_date).days)

    return (
        statistics.median(date_differences),
        stats.median_abs_deviation(date_differences)
    )


def _compute_realism_of_introduction(
    project_name: str, rawbugs: tp.FrozenSet[RawBug]
) -> tp.Tuple[float, float]:
    """
    Computes the median time difference and median absolute deviation between
    the introductions of the same fix. Result determines the realism of bug
    introduction.

    Args:
        rawbugs: The set of bugs to analyze

    Returns:
        A tuple (median, median absolute deviation)
    """
    project_repo = get_local_project_git(project_name)
    date_differences: tp.List[int] = []
    for rawbug in rawbugs:
        for intro_hash_a in rawbug.introducing_commits:
            introducer_a = project_repo.revparse_single(intro_hash_a)
            introducer_a_date = datetime.fromtimestamp(introducer_a.commit_time)
            for intro_hash_b in pybug.introducing_commits:
                introducer_b = project_repo.revparse_single(intro_hash_b)
                introducer_b_date = datetime.fromtimestamp(
                    introducer_b.commit_time
                )

                date_differences.append(
                    abs(introducer_a_date - introducer_b_date).days
                )

    return (
        statistics.median(date_differences),
        stats.median_abs_deviation(date_differences)
    )
