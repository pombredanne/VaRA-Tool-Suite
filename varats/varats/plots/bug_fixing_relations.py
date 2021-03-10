import typing as tp

import numpy as np
import plotly.graph_objs as gob
import plotly.offline as offply
import pygit2

from varats.data.reports.szz_report import SZZUnleashedReport
from varats.plot.plot import Plot
from varats.project.project_util import (
    get_project_cls_by_name,
    get_local_project_git,
)
from varats.provider.bug.bug import RawBug
from varats.provider.bug.bug_provider import BugProvider
from varats.revision.revisions import get_processed_revisions_files


def _plot_chord_diagram_for_raw_bugs(
    project_name: str, bug_set: tp.FrozenSet[RawBug]
) -> gob.Figure:
    """Creates a chord diagram representing relations between introducing/fixing
    commits for a given set of RawBugs."""
    project_repo = get_local_project_git(project_name)

    commits_to_nodes_map = _map_commits_to_nodes(project_repo)
    commit_count = len(commits_to_nodes_map.keys())
    commit_coordinates = _compute_node_placement(commit_count)

    edge_colors = ['#d4daff', '#84a9dd', '#5588c8', '#6d8acf']
    node_color = 'rgba(0,51,181, 0.85)'
    init_color = 'rgba(207, 0, 15, 1)'

    lines = []
    nodes = []
    for bug in bug_set:
        bug_fix = bug.fixing_commit
        fix_ind = commits_to_nodes_map[bug_fix]
        fix_coordinates = commit_coordinates[fix_ind]

        for bug_introduction in bug.introducing_commits:
            intro_ind = commits_to_nodes_map[bug_introduction]
            intro_coordinates = commit_coordinates[intro_ind]

            # get distance between the points and the respective interval index
            dist = _get_distance(fix_coordinates, intro_coordinates)
            interval = _get_interval(dist)
            color = edge_colors[interval]

            lines.append(
                _create_line(
                    fix_coordinates, intro_coordinates, color,
                    f'introduced by {bug_introduction}'
                )
            )

        # add fixing commits as vertices
        nodes.append(
            _create_node(fix_coordinates, node_color, f'bug fix: {bug_fix}')
        )

    init = _create_node(commit_coordinates, init_color, "HEAD")
    data = lines + nodes + [init]
    layout = _create_layout(f'Bug fixing relations for {project_name}')
    return gob.Figure(data=data, layout=layout)


def _bug_data_diff_plot(
    project_name: str, bugs_a: tp.FrozenSet[RawBug],
    bugs_b: tp.FrozenSet[RawBug]
) -> gob.Figure:
    project_repo = get_local_project_git(project_name)

    commits_to_nodes_map = _map_commits_to_nodes(project_repo)
    commit_count = len(commits_to_nodes_map.keys())
    commit_coordinates = _compute_node_placement(commit_count)

    init_color = 'rgba(207, 0, 15, 1)'
    node_color_default = 'rgba(0,51,181, 0.85)'
    node_color_a = "#ff0000"
    node_color_b = "#00ff00"
    edge_color_a = "#ff5555"
    edge_color_b = "#55ff55"

    lines: tp.List[gob.Scatter] = []
    nodes: tp.List[gob.Scatter] = []
    for revision, diff_a, diff_b in _diff_raw_bugs(bugs_a, bugs_b):
        bug_fix = revision
        fix_ind = commits_to_nodes_map[bug_fix]
        fix_coordinates = commit_coordinates[fix_ind]

        if diff_a:
            for introducer in diff_a:
                lines.append(
                    _create_line(
                        fix_coordinates,
                        commit_coordinates[commits_to_nodes_map[introducer]],
                        edge_color_a, f'introduced by {introducer}'
                    )
                )
        if diff_b:
            for introducer in diff_b:
                lines.append(
                    _create_line(
                        fix_coordinates,
                        commit_coordinates[commits_to_nodes_map[introducer]],
                        edge_color_b, f'introduced by {introducer}'
                    )
                )

        node_color = node_color_default
        if diff_a is None and diff_b is not None:
            node_color = node_color_b
        if diff_b is None and diff_a is not None:
            node_color = node_color_a
        if diff_a is None and diff_b is None:
            node_color = "#ffff00"

        nodes.append(
            _create_node(fix_coordinates, node_color, f'bug fix: {bug_fix}')
        )

    init = _create_node(commit_coordinates, init_color, "HEAD")
    data = lines + nodes + [init]
    layout = _create_layout(f'SZZ diff {project_name}')
    return gob.Figure(data=data, layout=layout)


KeyT = tp.TypeVar("KeyT")

ValueT = tp.TypeVar("ValueT")


def _get_distance(p1: tp.List[int], p2: tp.List[int]) -> float:
    # Returns distance between two points
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


__cp_parameters = [1.2, 1.5, 1.8, 2.1]

__distance_thresholds = [
    0.0,
    _get_distance([1, 0], 2 * [np.sqrt(2) / 2]),
    np.sqrt(2),
    _get_distance([1, 0], [-np.sqrt(2) / 2, np.sqrt(2) / 2]), 2.0
]


def _create_line(
    start: np.array, end: np.array, color: str, annotation: str
) -> gob.Scatter:
    dist = _get_distance(start, end)
    interval = _get_interval(dist)
    control_points = [
        start, start / __cp_parameters[interval],
        end / __cp_parameters[interval], end
    ]
    curve_points = _get_bezier_curve(control_points)

    return gob.Scatter(
        x=curve_points[:, 0],
        y=curve_points[:, 1],
        mode='lines',
        line=dict(color=color, shape='spline'),
        text=annotation,
        hoverinfo='text'
    )


def _create_node(coordinates: np.array, color: str, text: str) -> gob.Scatter:
    return gob.Scatter(
        x=[coordinates[0]],
        y=[coordinates[1]],
        mode='markers',
        name='',
        marker=dict(symbol='circle', size=8, color=color),
        text=text,
        hoverinfo='text'
    )


def _create_layout(title: str) -> gob.Layout:
    axis = dict(
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title=''
    )  # hide the axis
    width = 900
    height = 900
    layout = gob.Layout(
        title=title,
        showlegend=False,
        autosize=False,
        width=width,
        height=height,
        xaxis=dict(axis),
        yaxis=dict(axis),
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return layout


def _diff_raw_bugs(
    bugs_a: tp.FrozenSet[RawBug], bugs_b: tp.FrozenSet[RawBug]
) -> tp.Generator[tp.Tuple[str, tp.Optional[tp.FrozenSet[str]],
                           tp.Optional[tp.FrozenSet[str]]], None, None]:
    for fixing_commit, introducers_a, introducers_b in _zip_dicts({
        bug.fixing_commit: bug.introducing_commits for bug in bugs_a
    }, {bug.fixing_commit: bug.introducing_commits for bug in bugs_b}):
        diff_a: tp.Optional[tp.FrozenSet[str]] = None
        diff_b: tp.Optional[tp.FrozenSet[str]] = None
        if introducers_a:
            diff_a = introducers_a
            if introducers_b:
                diff_a = introducers_a.difference(introducers_b)
        if introducers_b:
            diff_b = introducers_b
            if introducers_a:
                diff_b = introducers_b.difference(introducers_a)

        yield fixing_commit, diff_a, diff_b


def _zip_dicts(
    a: tp.Dict[KeyT, ValueT], b: tp.Dict[KeyT, ValueT]
) -> tp.Generator[tp.Tuple[KeyT, tp.Optional[ValueT], tp.Optional[ValueT]],
                  None, None]:
    for i in a.keys() | b.keys():
        yield i, a.get(i, None), b.get(i, None)


def _get_bezier_curve(ctrl_points: np.array, num_points: int = 5) -> np.array:
    """Implements bezier edges to display between commit nodes."""
    n = len(ctrl_points)

    def get_coordinate_on_curve(factor: float) -> np.array:
        points_cp = np.copy(ctrl_points)
        for r in range(1, n):
            points_cp[:n - r, :] = (
                1 - factor
            ) * points_cp[:n - r, :] + factor * points_cp[1:n - r + 1, :]
        return points_cp[0, :]

    point_space = np.linspace(0, 1, num_points)
    return np.array([
        get_coordinate_on_curve(point_space[k]) for k in range(num_points)
    ])


def _get_interval(distance: float) -> int:
    # get right interval for given distance using distance thresholds
    # interval indices are in [0,3] for 5 thresholds
    k = 0
    while __distance_thresholds[k] < distance:
        k += 1
    return k - 1


def _compute_node_placement(commit_count: int) -> tp.List[np.array]:
    """Compute unit circle coordinates for each commit; move unit circle such
    that HEAD is on top."""
    theta_vals = np.linspace(-3 * np.pi / 2, np.pi / 2, commit_count)
    commit_coordinates: tp.List[np.array] = list()
    for theta in theta_vals:
        commit_coordinates.append(np.array([np.cos(theta), np.sin(theta)]))
    return commit_coordinates


def _map_commits_to_nodes(project_repo: pygit2.Repository) -> tp.Dict[str, int]:
    """Maps commit hex -> node id."""
    commits_to_nodes_map: tp.Dict[str, int] = {}
    commit_count = 0
    for commit in project_repo.walk(
        project_repo.head.target.hex, pygit2.GIT_SORT_TIME
    ):
        # node ids are sorted by time
        commits_to_nodes_map[commit.hex] = commit_count
        commit_count += 1
    return commits_to_nodes_map


class BugFixingRelationPlot(Plot):
    """Plot showing which commit fixed a bug introduced by which commit."""

    NAME = 'bug_relation_graph'

    def __init__(self, **kwargs: tp.Any) -> None:
        super().__init__(self.NAME, **kwargs)

    @staticmethod
    def supports_stage_separation() -> bool:
        return False

    def plot(self, view_mode: bool) -> None:
        """Plots bug plot for the whole project."""
        project_name = self.plot_kwargs["project"]

        bug_provider = BugProvider.get_provider_for_project(
            get_project_cls_by_name(project_name)
        )

        pydriller_bugs = bug_provider.find_all_raw_bugs()
        reports = get_processed_revisions_files(
            project_name, SZZUnleashedReport
        )
        szzunleashed_bugs = SZZUnleashedReport(reports[0]).get_all_raw_bugs()

        figure = _plot_chord_diagram_for_raw_bugs(project_name, pydriller_bugs)
        if view_mode:
            figure.show()
        else:
            offply.plot(
                figure, filename="pydriller_" + self.plot_file_name("html")
            )

        figure = _plot_chord_diagram_for_raw_bugs(
            project_name, szzunleashed_bugs
        )
        if view_mode:
            figure.show()
        else:
            offply.plot(
                figure, filename="szzunleashed_" + self.plot_file_name("html")
            )

        figure = _bug_data_diff_plot(
            project_name, pydriller_bugs, szzunleashed_bugs
        )
        if view_mode:
            figure.show()
        else:
            offply.plot(figure, filename="diff_" + self.plot_file_name("html"))

    def calc_missing_revisions(self, boundary_gradient: float) -> tp.Set[str]:
        return set()
