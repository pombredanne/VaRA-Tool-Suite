import typing as tp
from pathlib import Path

import yaml

from varats.base.version_header import VersionHeader
from varats.provider.bug.bug import RawBug
from varats.report.report import BaseReport, FileStatusExtension, MetaReport


class SZZReport(BaseReport):
    """
    Base class for reports created by several SZZ tools.

    Subclasses of this report only differ in the tool used and the shorthand.
    """

    SHORTHAND = "SZZ"
    FILE_TYPE = "yaml"

    def __init__(self, path: Path, szz_tool: str):
        super().__init__(path)
        self.__path = path
        with open(path, 'r') as stream:
            documents = yaml.load_all(stream, Loader=yaml.CLoader)
            version_header = VersionHeader(next(documents))
            version_header.raise_if_not_type("SZZReport")
            version_header.raise_if_version_is_less_than(1)
            raw_report = next(documents)
            if not raw_report["szz_tool"] == szz_tool:
                raise AssertionError(
                    "Report was not created with the correct tool."
                )
            self.__bugs: tp.Dict[str, RawBug] = {}
            for fix, introducers in raw_report["bugs"].items():
                self.__bugs[fix] = RawBug(fix, set(introducers), None)

    def get_all_raw_bugs(self) -> tp.FrozenSet[RawBug]:
        """
        Get the set of all bugs in this report.

        Returns:
            A set of `RawBug` s.
        """
        return frozenset(self.__bugs.values())

    def get_raw_bug_by_fix(self, fixing_commit: str) -> tp.Optional[RawBug]:
        """
        Get a bug by the id of the fixing commit.

        Returns:
            A `RawBug` if avilable, else `None`.
        """
        return self.__bugs.get(fixing_commit, None)


class SZZUnleashedReport(SZZReport):
    SHORTHAND = "SZZU"

    def __init__(self, path: Path):
        super().__init__(path, "SZZUnleashed")

    @staticmethod
    def get_file_name(
        project_name: str,
        binary_name: str,
        project_version: str,
        project_uuid: str,
        extension_type: FileStatusExtension,
        file_ext: str = "yaml"
    ) -> str:
        """
        Generates a filename for a commit report with 'yaml' as file extension.

        Args:
            project_name: name of the project for which the report was generated
            binary_name: name of the binary for which the report was generated
            project_version: version of the analyzed project, i.e., commit hash
            project_uuid: benchbuild uuid for the experiment run
            extension_type: to specify the status of the generated report
            file_ext: file extension of the report file

        Returns:
            name for the report file that can later be uniquly identified
        """
        return MetaReport.get_file_name(
            SZZUnleashedReport.SHORTHAND, project_name, binary_name,
            project_version, project_uuid, extension_type, file_ext
        )


class OpenSZZReport(SZZReport):
    SHORTHAND = "OSZZ"

    def __init__(self, path: Path):
        super().__init__(path, "OpenSZZ")

    @staticmethod
    def get_file_name(
        project_name: str,
        binary_name: str,
        project_version: str,
        project_uuid: str,
        extension_type: FileStatusExtension,
        file_ext: str = "yaml"
    ) -> str:
        """
        Generates a filename for a commit report with 'yaml' as file extension.

        Args:
            project_name: name of the project for which the report was generated
            binary_name: name of the binary for which the report was generated
            project_version: version of the analyzed project, i.e., commit hash
            project_uuid: benchbuild uuid for the experiment run
            extension_type: to specify the status of the generated report
            file_ext: file extension of the report file

        Returns:
            name for the report file that can later be uniquly identified
        """
        return MetaReport.get_file_name(
            OpenSZZReport.SHORTHAND, project_name, binary_name, project_version,
            project_uuid, extension_type, file_ext
        )
