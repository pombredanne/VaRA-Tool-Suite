"""Base table module."""

import abc
import logging
import typing as tp
from pathlib import Path

from pylatex import Document, Package, NoEscape, UnsafeCommand

from varats.paper.case_study import CaseStudy
from varats.table.tables import TableFormat, TableConfig

LOG = logging.getLogger(__name__)


class TableDataEmpty(Exception):
    """Throw if there was no input for the table."""


class Table:
    """An abstract base class for all tables generated by VaRA-TS."""

    NAME = "Table"
    TABLES: tp.Dict[str, tp.Type['Table']] = {}

    format_filetypes = {
        TableFormat.GITHUB: "md",
        TableFormat.HTML: "html",
        TableFormat.UNSAFEHTML: "html",
        TableFormat.LATEX: "tex",
        TableFormat.LATEX_RAW: "tex",
        TableFormat.LATEX_BOOKTABS: "tex",
        TableFormat.RST: "rst",
    }

    def __init__(
        self, name: str, table_config: TableConfig, **kwargs: tp.Any
    ) -> None:
        self.__name = name
        self.__table_config = table_config
        self.__saved_extra_args = kwargs

    @classmethod
    def __init_subclass__(cls, **kwargs: tp.Any) -> None:
        """Register concrete tables."""
        super().__init_subclass__(**kwargs)

    @staticmethod
    def get_table_types_help_string() -> str:
        """
        Generates help string for visualizing all available tables.

        Returns:
            a help string that contains all available table names.
        """
        return "The following tables are available:\n  " + "\n  ".join(
            list(Table.TABLES)
        )

    @staticmethod
    def get_class_for_table_type(table_type: str) -> tp.Type['Table']:
        """
        Get the class for table from the table registry.

        Args:
            table_type: The name of the table.

        Returns: The class implementing the table.
        """
        if table_type not in Table.TABLES:
            raise LookupError(
                f"Unknown table '{table_type}'.\n" +
                Table.get_table_types_help_string()
            )

        table_cls = Table.TABLES[table_type]
        return table_cls

    @property
    def name(self) -> str:
        """
        Name of the current table.

        Test:
        >>> Table('test', TableConfig.from_kwargs(view=False)).name
        'test'
        """
        return self.__name

    @property
    def table_config(self) -> TableConfig:
        """Table config for this table."""
        return self.__table_config

    @property
    def table_kwargs(self) -> tp.Any:
        """
        Access the kwargs passed to the initial table.

        Test:
        >>> p = Table('test', TableConfig.from_kwargs(view=False), foo='bar', \
                     baz='bazzer')
        >>> p.table_kwargs['foo']
        'bar'
        >>> p.table_kwargs['baz']
        'bazzer'
        """
        return self.__saved_extra_args

    @staticmethod
    def supports_stage_separation() -> bool:
        """True, if the table supports stage separation, i.e., the table can be
        drawn separating the different stages in a case study."""
        return False

    @abc.abstractmethod
    def tabulate(self) -> str:
        """Build the table using tabulate."""

    def table_file_name(self) -> str:
        """
        Get the file name this table; will be stored to when calling save.
        Automatically deduces this tables' filetype from its format.

        Returns:
            the file name the table will be stored to

        Test:
        >>> p = Table('test', TableConfig.from_kwargs(view=False), project='bar')
        >>> p.table_file_name('txt')
        'bar_test.txt'
        >>> from varats.paper.case_study import CaseStudy
        >>> p = Table('foo', TableConfig.from_kwargs(view=False), project='bar', \
                     case_study=CaseStudy('baz', 42))
        >>> p.table_file_name('tex')
        'baz_42_foo.tex'
        """
        # TODO: Change file name to sth. unique of each case study. This should
        #       allow us to use the REQUIRE_MULTI_CASE_STUDY for most tables
        #       without instantly overwriting the generated table with the next
        #       one.
        table_ident = ''
        if 'case_study' in self.table_kwargs:
            cs: tp.Union[CaseStudy,
                         tp.List[CaseStudy]] = self.table_kwargs['case_study']

            if isinstance(cs, list) and len(cs) == 1:
                cs = cs.pop(0)

            if isinstance(cs, CaseStudy):
                table_ident = f"{cs.project_name}_{cs.version}_"

        sep_stages = ''
        if self.supports_stage_separation(
        ) and self.table_kwargs.get('sep_stages', None):
            sep_stages = 'S'

        filetype: str = self.format_filetypes.get(
            self.table_kwargs["format"], "txt"
        )
        return f"{table_ident}{self.name}{sep_stages}.{filetype}"

    @abc.abstractmethod
    def wrap_table(self, table: str) -> str:
        """
        Used to wrap tables inside a complete latex document by passing desired
        parameters to wrap_table_in_document.

        Returns:
            The resulting table string.
        """

    def show(self) -> None:
        """Show the current table in console."""
        try:
            table = self.tabulate()
        except TableDataEmpty:
            LOG.warning("No data for the current project.")
            return
        print(table)

    def save(self, path: Path, wrap_document: bool) -> None:
        """
        Save the current table to a file.

        Args:
            path: the path where the file is stored (excluding file name)
            wrap_document: if enabled wraps the given table inside a proper
                           latex document
        """
        try:
            table = self.tabulate()
        except TableDataEmpty:
            LOG.warning("No data for this table.")
            return

        if wrap_document:
            table = self.wrap_table(table)

        with open(path / self.table_file_name(), "w") as outfile:
            outfile.write(table)


def wrap_table_in_document(
    table: str, landscape: bool = False, margin: float = 1.5
) -> str:
    """
    Wraps given table inside a proper latex document. Uses longtable instead of
    tabular to fit data on multiple pages.

    Args:
        table: table string to wrap the document around.
        landscape: orientation of the table document. True for landscape mode,
                   i.e. horizontal orientation.
        margin: margin of the wrapped table inside the resulting document.

    Returns:
        string representation of the resulting latex document.
    """
    doc = Document(
        documentclass="scrbook",
        document_options="paper=a4",
        geometry_options={
            "margin": f"{margin}cm",
            "landscape": "true" if landscape else "false"
        }
    )
    # set monospace font
    monospace_comm = UnsafeCommand(
        'renewcommand', r'\familydefault', extra_arguments=r'\ttdefault'
    )
    doc.preamble.append(monospace_comm)

    # package in case longtables are used
    doc.packages.append(Package('longtable'))
    # package for booktabs automatically generated by pandas.to_latex()
    doc.packages.append(Package('booktabs'))

    doc.change_document_style("empty")

    # embed latex table inside document
    doc.append(NoEscape(table))

    # dump function returns string representation of document
    return tp.cast(str, doc.dumps())
