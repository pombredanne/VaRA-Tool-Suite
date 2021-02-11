"""
OSV API module.

https://osv.dev/docs
"""
import json
import logging
import typing as tp
from enum import Enum

from benchbuild.utils.cmd import curl
from benchbuild.utils.revision_ranges import (
    RevisionRange,
    AbstractRevisionRange,
    SingleRevision,
)

LOG = logging.getLogger(__name__)


class OSVPackageInfo():
    """Identifies a OSV package/project."""

    def __init__(self, name: str, ecosystem: tp.Optional[str] = None):
        self.__name = name
        self.__ecosystem = ecosystem

    @property
    def name(self) -> str:
        """
        The name of the package.

        Returns:
            the name of the package
        """
        return self.__name

    @property
    def ecosystem(self) -> tp.Optional[str]:
        """
        The ecosystem of the package. This is None for C/C++ projects.

        Returns:
            the ecosystem of the package
        """
        return self.__ecosystem

    def as_dict(self):
        """
        Return a dictionary representation of this object.

        Returns:
            a dictionary representation of this object
        """
        result = {"name": self.name}
        if self.ecosystem:
            result["ecosystem"] = self.ecosystem
        return result


class OSVRange():
    """
    A range of commits as used by OSV.

    The range is defined by an introducing and a fixing commit (range). Both,
    the introducing and the fixing commitss can be either a single commit or a
    short range of commits.
    """

    def __init__(
        self, introduced_in: AbstractRevisionRange,
        fixed_in: AbstractRevisionRange
    ):
        self.__introduced_in: introduced_in
        self.__fixed_in: fixed_in

    @property
    def introduced_in(self) -> AbstractRevisionRange:
        """
        The introducing commit (range).

        Returns:
            the introducing commit (range).
        """
        return self.introduced_in

    @property
    def fixed_in(self) -> AbstractRevisionRange:
        """
        The fixing commit (range).

        Returns:
            the fixing commit (range).
        """
        return self.fixed_in


class OSVSeverity(Enum):
    """Severity levels used by OSV vulnerabilities."""
    NONE = "NONE",
    LOW = "LOW",
    MEDIUM = "MEDIUM",
    HIGH = "HIGH",
    CRITICAL = "CRITICAL"


class OSVVulnerability():
    """A vulnerability as described by OSV."""

    def __init__(
        self, id: str, package: OSVPackageInfo, summary: str, details: str,
        severity: OSVSeverity, affected_versions: tp.Set[str],
        affected_ranges: tp.Set[OSVRange], reference_urls: tp.List[str],
        cves: tp.List[str]
    ):
        self.__id = id
        self.__package: OSVPackageInfo = package
        self.__summary = summary
        self.__details = details
        self.__severity = severity
        self.__affected_versions = affected_versions
        self.__affected_ranges = affected_ranges
        self.__reference_urls = reference_urls
        self.__cves = cves

    @property
    def id(self) -> str:
        """
        Unique identifier for this vulnerability (assigned by OSV). This is of
        the format YEAR-N (e.g. "2020-111").

        Returns:
            unique identifier for this vulnerability
        """
        return self.__id

    @property
    def package(self) -> OSVPackageInfo:
        """
        Package information.

        Returns:
            package information
        """
        return self.__package

    @property
    def summary(self) -> str:
        """
        One line human readable summary for the vulnerability.

        Returns:
            one line human readable summary for the vulnerability
        """
        return self.__summary

    @property
    def details(self) -> str:
        """
        Additional human readable details for the vulnerability.

        Returns:
            additional human readable details for the vulnerability
        """
        return self.__details

    @property
    def severity(self) -> OSVSeverity:
        """
        Severity of the vulnerability.

        Returns:
            the severity of the vulnerability
        """
        return self.__severity

    @property
    def affected_versions(self) -> tp.Set[str]:
        """
        List of affected versions. This should match tag names in the upstream
        repository.

        Returns:
            list of affected versions
        """
        return self.__affected_versions

    @property
    def affected_ranges(self) -> tp.Set[OSVRange]:
        """
        The commit ranges that contain this vulnerability. Each range entry
        should represent a different upstream branch.

        Returns:
            the commit ranges that contain this vulnerability
        """
        return self.__affected_ranges

    @property
    def reference_rls(self) -> tp.List[str]:
        """
        URLs to more information/advisories.

        Returns:
            URLs to more information/advisories
        """
        return self.__reference_urls

    @property
    def cves(self) -> tp.List[str]:
        """
        CVEs, if allocated.

        Returns:
            CVEs, if allocated
        """
        return self.__cves

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, OSVVulnerability):
            return False
        return self.id == o.id

    def __hash__(self) -> int:
        return hash(self.id)


__QUERY_URL = "https://api.osv.dev/v1/query?key={api_key}"


def __parse_osv_commit(commit: str) -> AbstractRevisionRange:
    if ":" in commit:
        return RevisionRange(*commit.split(":", maxsplit=1))
    return SingleRevision(commit)


OSVRangeJSON = tp.List[tp.Dict[str, tp.Dict[str, str]]]


def __parse_osv_ranges(range_obj: OSVRangeJSON) -> tp.Set[OSVRange]:
    ranges: tp.Set[OSVRange] = set()
    for entry in range_obj:
        introduced_json = entry["introducedIn"]
        fixed_json = entry["fixedIn"]

        ranges.add(
            OSVRange(
                __parse_osv_commit(introduced_json["commit"]),
                __parse_osv_commit(fixed_json["commit"])
            )
        )
    return ranges


def __osv_api_query_vulnerabilities(
    package: OSVPackageInfo,
    commit: tp.Optional[str] = None,
    version: tp.Optional[str] = None
) -> tp.List[OSVVulnerability]:
    """
    Query vulnerabilities for a particular project at a given commit or version.

    Args:
        commit: the commit hash to query for.
                If specified, "version" should not be set.
        version: the version string to query for.
                 If specified, "commit" should not be set.
        package: the package to query against.

    Returns:
        a list of vulnerabilities for the given commit/version of the package.
    """
    if commit and version:
        raise AssertionError("Cannot use commit and version at the same time.")
    if not commit and not version:
        raise AssertionError("Must specify either commit or version")

    # todo: get api key
    api_key: str = ""

    payload = {"package": package.as_dict()}
    if commit:
        payload["commit"] = commit
    if version:
        payload["version"] = version

    payload_str = json.dumps(payload)

    result = curl["-X", "POST", "-d", f"'{payload_str}'",
                  __QUERY_URL.format(api_key=api_key)]
    result_json: tp.Dict[str, tp.Any] = json.loads(result)
    if "vulns" not in result_json.keys():
        LOG.warning(
            f"Error response from OSV API: "
            f"(error code {result_json.get('code', '?')}) "
            f"{result_json.get('message', '')}"
        )
        return []

    vulnerabilities: tp.List[OSVVulnerability] = []
    for entry in result_json["vulns"]:
        vulnerabilities.append(
            OSVVulnerability(
                entry["id"],
                OSVPackageInfo(
                    entry["package"]["name"],
                    entry["package"].get("ecosystem", None)
                ), entry["summary"], entry["details"],
                OSVSeverity.get(entry["severity"], OSVSeverity.NONE),
                set(entry["affects"].get("versions", [])),
                __parse_osv_ranges(entry["affects"]["ranges"]),
                entry.get("referenceUrls", None), entry.get("cves", None)
            )
        )

    return vulnerabilities
