"""Module for all reports generated for taint flow analyses."."""

from varats.report.report import BaseReport


class TaintPropagationReport(BaseReport, shorthand="TPR", file_type="txt"):
    """Print the result of filechecking a llvm ir file generated by VaRA or
    Phasar in a readable manner."""

    def __repr__(self) -> str:
        return self.shorthand() + ": " + self.path.name

    def __lt__(self, other: 'TaintPropagationReport') -> bool:
        return self.path < other.path
