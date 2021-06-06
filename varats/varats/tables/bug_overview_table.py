"""Module for writing bug-data metrics tables."""
import statistics
import typing as tp
from datetime import date, datetime

import numpy as np
import pandas as pd
import pygit2
from tabulate import tabulate

from varats.project.project_util import (
    get_project_cls_by_name,
    get_local_project_git,
)
from varats.provider.bug.bug import RawBug, PygitBug
from varats.provider.bug.bug_provider import BugProvider
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
        rawbugs = bug_provider.find_all_raw_bugs()

        data = [
            _compute_fixing_evaluation_row(
                project_name, start_date, rawbugs, input_fixing_commits
            )
        ]

        eval_df = pd.DataFrame(data=np.array(data), columns=variables)

        if self.format in [
            TableFormat.latex, TableFormat.latex_raw, TableFormat.latex_booktabs
        ]:
            tex_code = eval_df.to_latex(bold_rows=True, multicolumn_format="c")
            return str(tex_code) if tex_code else ""
        return tabulate(eval_df, eval_df.columns, self.format.value)

    def wrap_table(self, table: str) -> str:
        return wrap_table_in_document(table=table, landscape=True)


class BugIntroducingEvaluationTable(Table):
    """Visualizes different metrics on introducing commits of bugs without
    ground truth."""

    NAME = "bug_introducing_evaluation"

    def __init__(self, **kwargs: tp.Any):
        super().__init__(self.NAME, **kwargs)

    def tabulate(self) -> str:
        project_name = self.table_kwargs["project"]

        bug_provider = BugProvider.get_provider_for_project(
            get_project_cls_by_name(project_name)
        )
        pybugs = bug_provider.find_all_pygit_bugs()

        variables = ["message bugs", "issue bugs", "realism of intro."]

        data = [_compute_introducing_evaluation_row(pybugs)]

        eval_df = pd.DataFrame(data=data, columns=variables)

        if self.format in [
            TableFormat.latex, TableFormat.latex_raw, TableFormat.latex_booktabs
        ]:
            tex_code = eval_df.to_latex(bold_rows=True, multicolumn_format="c")
            return str(tex_code) if tex_code else ""
        return tabulate(eval_df, eval_df.columns, self.format.value)

    def wrap_table(self, table: str) -> str:
        return wrap_table_in_document(table=table, landscape=True)


def _compute_fixing_evaluation_row(
    project_name: str, start_date: datetime, rawbugs: tp.FrozenSet[RawBug],
    input_fixing_commits: tp.Set[str]
) -> tp.List[int]:
    """
    Format: commits total, true fixes, fixes found, tp, fp, tn, fn
    """
    project_repo = get_local_project_git(project_name)

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
    commit_count = 0
    found_non_fixing_commits: tp.Set[str] = set()
    for commit in project_repo.walk(
        project_repo.head.target.hex, pygit2.GIT_SORT_TIME
    ):
        if datetime.fromtimestamp(commit.commit_time) < start_date:
            break
        if commit.hex not in found_fixing_commits:
            found_non_fixing_commits.add(commit.hex)
        commit_count = commit_count + 1

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

    return [
        commit_count,
        len(true_fixing_commits),
        len(found_fixing_commits),
        len(tp_commits),
        len(fp_commits),
        len(tn_commits),
        len(fn_commits)
    ]


def _compute_introducing_evaluation_row(
    pybugs: tp.FrozenSet[PygitBug]
) -> tp.List[int]:
    """
    Format: commit message bugs, issue event bugs, realism of bug introduction
    (in days)
    """
    message_bugs = 0
    issue_bugs = 0
    date_differences: tp.List[int] = []
    for pybug in pybugs:
        if pybug.issue_id:
            issue_bugs = issue_bugs + 1
        else:
            message_bugs = message_bugs + 1

        fixing_date = datetime.fromtimestamp(pybug.fixing_commit.commit_time)
        for introducer in pybug.introducing_commits:
            intro_date = datetime.fromtimestamp(introducer.commit_time)
            date_differences.append((fixing_date - intro_date).days)

    return [message_bugs, issue_bugs, statistics.median(date_differences)]
