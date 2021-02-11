"""Module for the :class:`BugProvider`."""
import logging
import sys
import typing as tp

from benchbuild.project import Project

from varats.provider.provider import Provider

if sys.version_info <= (3, 8):
    from typing_extensions import Protocol
    from typing_extensions import runtime_checkable
else:
    from typing import Protocol
    from typing import runtime_checkable

LOG = logging.getLogger(__name__)


@runtime_checkable
class OSVProviderHook(Protocol):
    """Gives the :class:`OSVProvider` the necessary information how to find the
    project in the OSV database."""

    @classmethod
    def get_osv_package_info(cls) -> tp.List[tp.Tuple[str, str]]:
        """
        Get the osv package info for a project.

        Returns:
            the project's osv package info
        """
        ...


class OSVProvider(Provider):
    """Provides bug information for a project."""

    def __init__(self, project: tp.Type[Project]) -> None:
        super().__init__(project)
        if isinstance(project, OSVProviderHook):
            self.hook = tp.cast(OSVProviderHook, project)
        else:
            raise ValueError(
                f"Project {project} does not implement "
                f"OSVProviderHook."
            )

    @classmethod
    def create_provider_for_project(
        cls, project: tp.Type[Project]
    ) -> tp.Optional['OSVProvider']:
        if isinstance(project, OSVProviderHook):
            return OSVProvider(project)
        return None

    @classmethod
    def create_default_provider(
        cls, project: tp.Type[Project]
    ) -> 'OSVProvider':
        return OSVDefaultProvider(project)


class OSVDefaultProvider(OSVProvider):
    """Default implementation of the :class:`OSVProvider` for projects that have
    no OSV package."""

    def __init__(self, project: tp.Type[Project]) -> None:
        # pylint: disable=E1003
        super(OSVProvider, self).__init__(project)
