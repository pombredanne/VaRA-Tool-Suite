"""Module for representing blame interaction data in a graph/network."""
import abc
import sys
import typing as tp

import networkx as nx

from varats.data.reports.blame_report import (
    BlameReport,
    gen_base_to_inter_commit_repo_pair_mapping,
    BlameReportDiff,
)
from varats.jupyterhelper.file import load_blame_report
from varats.plot.plot import PlotDataEmpty
from varats.revision.revisions import get_processed_revisions_files
from varats.utils.git_util import (
    CommitRepoPair,
    create_commit_lookup_helper,
    DUMMY_COMMIT,
)

if sys.version_info <= (3, 8):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class _BIGNodeAttrs(TypedDict):
    """Blame interaction graph node attributes."""
    commit: CommitRepoPair


class _BIGEdgeAttrs(TypedDict):
    """Blame interaction graph edge attributes."""
    amount: int


class CIGNodeAttrs(TypedDict):
    """Commit interaction graph node attributes."""
    commit: CommitRepoPair


class CIGEdgeAttrs(TypedDict):
    """Commit interaction graph edge attributes."""
    amount: int


class AIGNodeAttrs(TypedDict):
    """Author interaction graph node attributes."""
    author: str
    num_commits: int


class AIGEdgeAttrs(TypedDict):
    """Author interaction graph edge attributes."""
    amount: int


class CAIGNodeAttrs(TypedDict):
    """Commit-author interaction graph node attributes."""
    commit: tp.Optional[CommitRepoPair]
    author: tp.Optional[str]


class CAIGEdgeAttrs(TypedDict):
    """Commit-author interaction graph edge attributes."""
    amount: int


class InteractionGraph(abc.ABC):
    """Graph/Network built from interaction data."""

    def __init__(self, project_name: str):
        self.__project_name = project_name

    @property
    def project_name(self):
        return self.__project_name

    @abc.abstractmethod
    def _interaction_graph(self) -> nx.DiGraph:
        pass

    def commit_interaction_graph(self) -> nx.DiGraph:
        """
        Return a digraph with commits as nodes and interactions as edges.

        The graph has the following attributes:
        Nodes:
          - commit: CommitRepoPair for this commit
        Edges:
          - amount: how often this interaction was found

        Returns:
            the commit interaction graph
        """
        ig = self._interaction_graph()

        def edge_data(
            b: tp.Set[CommitRepoPair], c: tp.Set[CommitRepoPair]
        ) -> CIGEdgeAttrs:
            assert len(b) == len(c) == 1, "Some node has more than one commit."
            return tp.cast(
                CIGEdgeAttrs, ig[next(iter(b))][next(iter(c))].copy()
            )

        def node_data(b: tp.Set[CommitRepoPair]) -> CIGNodeAttrs:
            assert len(b) == 1, "Some node has more than one commit."
            return tp.cast(CIGNodeAttrs, ig.nodes[next(iter(b))].copy())

        return nx.quotient_graph(
            ig,
            partition=lambda u, v: False,
            edge_data=edge_data,
            node_data=node_data,
            # Use relabel=True so users cannot rely on the node type
            # but only on the graph attributes
            relabel=True,
            create_using=nx.DiGraph
        )

    def author_interaction_graph(self) -> nx.DiGraph:
        """
        Return a digraph with authors as nodes and interactions as edges.

        The graph has the following attributes:
        Nodes:
          - author: name of the author
          - num_commits: number of commits aggregated in this node
        Edges:
          - amount: how often an interaction between two authors was found

        Returns:
            the author interaction graph
        """
        ig = self._interaction_graph()
        commit_lookup = create_commit_lookup_helper(self.project_name)

        def partition(u: CommitRepoPair, v: CommitRepoPair) -> bool:
            if u.commit_hash == DUMMY_COMMIT or v.commit_hash == DUMMY_COMMIT:
                return u.commit_hash == v.commit_hash
            return str(commit_lookup(u).author.name
                      ) == str(commit_lookup(v).author.name)

        def edge_data(
            b: tp.Set[CommitRepoPair], c: tp.Set[CommitRepoPair]
        ) -> AIGEdgeAttrs:
            amount = 0
            for source in b:
                for sink in c:
                    if ig.has_edge(source, sink):
                        amount += int(ig[source][sink]["amount"])

            return {
                "amount": amount,
            }

        def node_data(b: tp.Set[CommitRepoPair]) -> AIGNodeAttrs:
            authors = {
                str(commit_lookup(commit).author.name)
                if commit.commit_hash != DUMMY_COMMIT else "Unknown"
                for commit in b
            }
            assert len(authors) == 1, "Some node has more then one author."
            return {
                "author": next(iter(authors)),
                "num_commits": len(b),
            }

        return nx.quotient_graph(
            ig,
            partition=partition,
            edge_data=edge_data,
            node_data=node_data,
            # Use relabel=True so users cannot rely on the node type
            # but only on the graph attributes
            relabel=True,
            create_using=nx.DiGraph
        )

    def commit_author_interaction_graph(
        self,
        outgoing_interactions: bool = True,
        incoming_interactions: bool = False
    ) -> nx.DiGraph:
        """
        Return a digraph connecting commits to interacting authors.

        The graph has the following attributes:
        Nodes:
          - commit: commit hash if the node is a commit
          - author: name of the author if the node is an author
        Edges:
          - amount: how often a commit interacts with an author

        Args:
            outgoing_interactions: whether to include outgoing interactions
            incoming_interactions: whether to include incoming interactions

        Returns:
            the commit-author interaction graph
        """
        ig = self._interaction_graph()
        commit_lookup = create_commit_lookup_helper(self.project_name)

        commit_author_mapping = {
            commit: commit_lookup(commit).author.name
            if commit.commit_hash != DUMMY_COMMIT else "Unknown"
            for commit in (list(ig.nodes))
        }
        caig = nx.DiGraph()
        # add commits as nodes
        caig.add_nodes_from([
            (commit, {
                "commit": ig.nodes[commit]["commit"],
                "author": None
            }) for commit in (list(ig.nodes))
        ])
        # add authors as nodes
        caig.add_nodes_from([(author, {
            "commit": None,
            "author": author
        }) for author in (set(commit_author_mapping.values()))])

        # add edges and aggregate edge attributes
        for node in ig.nodes:
            if incoming_interactions:
                for source, sink, data in ig.in_edges(node, data=True):
                    if not caig.has_edge(source, commit_author_mapping[sink]):
                        caig.add_edge(
                            source, commit_author_mapping[sink], amount=0
                        )
                    caig[source][commit_author_mapping[sink]
                                ]["amount"] += data["amount"]
            if outgoing_interactions:
                for source, sink, data in ig.out_edges(node, data=True):
                    if not caig.has_edge(sink, commit_author_mapping[source]):
                        caig.add_edge(
                            sink, commit_author_mapping[source], amount=0
                        )
                    caig[sink][commit_author_mapping[source]
                              ]["amount"] += data["amount"]

        # Relabel nodes to integers so users cannot rely on the node type
        # but only on the graph attributes
        return nx.convert_node_labels_to_integers(caig)


def create_blame_interaction_graph(
    project_name: str, revision: str
) -> BlameInteractionGraph:
    """
    Create a blame interaction graph for a certain project revision.

    Args:
        project_name: name of the project
        revision: project revision

    Returns:
        the blame interaction graph
    """
    file_name_filter: tp.Callable[[str], bool] = lambda x: False

    if revision:

        def match_revision(rev: str) -> bool:
            return True if rev == revision else False

        file_name_filter = match_revision

    report_files = get_processed_revisions_files(
        project_name, BlameReport, file_name_filter
    )
    if len(report_files) == 0:
        raise PlotDataEmpty(f"Found no BlameReport for project {project_name}")
    report = load_blame_report(report_files[0])
    return BlameInteractionGraph(project_name, report)


AttrType = tp.TypeVar("AttrType")
