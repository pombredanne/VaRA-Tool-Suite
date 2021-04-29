"""Base plot module."""

import abc
import logging
import typing as tp
from pathlib import Path

import matplotlib.pyplot as plt

if tp.TYPE_CHECKING:
    from varats.paper.case_study import CaseStudy  # pylint: disable=W0611

LOG = logging.getLogger(__name__)


class PlotDataEmpty(Exception):
    """Throw if there was no input data for plotting."""


class Plot():
    """An abstract base class for all plots generated by VaRA-TS."""

    PLOTS: tp.Dict[str, tp.Type['Plot']] = {}

    def __init__(self, name: str, **kwargs: tp.Any) -> None:
        self.__name = name
        self.__style = "classic"
        self.__saved_extra_args = kwargs

    @classmethod
    def __init_subclass__(cls, plot_name: str, **kwargs: tp.Any) -> None:
        """
        Register concrete plot generators.

        Args:
            generator_name: name for the plot generator as will be used in the
                            CLI interface
            plot:           plot class used by the generator
            options:        command line options needed by the generator
        """
        # mypy does not yet fully understand __init_subclass__()
        # https://github.com/python/mypy/issues/4660
        super().__init_subclass__(**kwargs)  # type: ignore
        cls.PLOTS[plot_name] = cls

    @staticmethod
    def get_plot_types_help_string() -> str:
        """
        Generates help string for visualizing all available plots.

        Returns:
            a help string that contains all available plot names.
        """
        return "The following plots are available:\n  " + "\n  ".join([
            key for key in Plot.PLOTS
        ])

    @staticmethod
    def get_class_for_plot_type(plot_type: str) -> tp.Type['Plot']:
        """
        Get the class for plot from the plot registry.

        Args:
            plot_type: The name of the plot.

        Returns: The class implementing the plot.
        """
        if plot_type not in Plot.PLOTS:
            raise LookupError(
                f"Unknown plot '{plot_type}'.\n" +
                Plot.get_plot_types_help_string()
            )

        plot_cls = Plot.PLOTS[plot_type]
        return plot_cls

    @property
    def name(self) -> str:
        """
        Name of the current plot.

        Test:
        >>> Plot('test').name
        'test'
        """
        return self.__name

    @property
    def style(self) -> str:
        """
        Current plot style.

        Test:
        >>> Plot('test').style
        'classic'
        """
        return self.__style

    @style.setter
    def style(self, new_style: str) -> None:
        """Access current style of the plot."""
        self.__style = new_style

    @property
    def plot_kwargs(self) -> tp.Any:
        """
        Access the kwargs passed to the initial plot.

        Test:
        >>> p = Plot('test', foo='bar', baz='bazzer')
        >>> p.plot_kwargs['foo']
        'bar'
        >>> p.plot_kwargs['baz']
        'bazzer'
        """
        return self.__saved_extra_args

    @staticmethod
    def supports_stage_separation() -> bool:
        """True, if the plot supports stage separation, i.e., the plot can be
        drawn separating the different stages in a case study."""
        return False

    @abc.abstractmethod
    def plot(self, view_mode: bool) -> None:
        """Plot the current plot to a file."""

    def show(self) -> None:
        """Show the current plot."""
        try:
            self.plot(True)
        except PlotDataEmpty:
            LOG.warning(f"No data for project {self.plot_kwargs['project']}.")
            return
        plt.show()
        plt.close()

    def plot_file_name(self, filetype: str) -> str:
        """
        Get the file name this plot; will be stored to when calling save.

        Args:
            filetype: the file type for the plot

        Returns:
            the file name the plot will be stored to

        Test:
        >>> p = Plot('test', project='bar')
        >>> p.plot_file_name('svg')
        'bar_test.svg'
        >>> from varats.paper.case_study import CaseStudy
        >>> p = Plot('foo', project='bar', plot_case_study=CaseStudy('baz', 42))
        >>> p.plot_file_name('png')
        'baz_42_foo.png'
        """
        plot_ident = ''
        if self.plot_kwargs.get('plot_case_study', None):
            case_study: 'CaseStudy' = self.plot_kwargs['plot_case_study']
            plot_ident = f"{case_study.project_name}_{case_study.version}_"
        elif 'project' in self.plot_kwargs:
            plot_ident = f"{self.plot_kwargs['project']}_"

        sep_stages = ''
        if self.supports_stage_separation(
        ) and self.plot_kwargs.get('sep_stages', None):
            sep_stages = 'S'

        return f"{plot_ident}{self.name}{sep_stages}.{filetype}"

    def save(self, plot_dir: Path, filetype: str = 'svg') -> None:
        """
        Save the current plot to a file.

        Args:
            plot_dir: the path where the file is stored(excluding the file name)
            filetype: the file type of the plot
        """
        try:
            self.plot(False)
        except PlotDataEmpty:
            LOG.warning(f"No data for project {self.plot_kwargs['project']}.")
            return

        # TODO (se-passau/VaRA#545): refactor dpi into plot_config.
        plt.savefig(
            plot_dir / self.plot_file_name(filetype),
            dpi=1200,
            bbox_inches="tight",
            format=filetype
        )
        plt.close()

    @abc.abstractmethod
    def calc_missing_revisions(self, boundary_gradient: float) -> tp.Set[str]:
        """
        Calculate a list of revisions that could improve precisions of this
        plot.

        Args:
            boundary_gradient: The maximal expected gradient in percent between
                               two revisions, every thing that exceeds the
                               boundary should be further analyzed.
        """
