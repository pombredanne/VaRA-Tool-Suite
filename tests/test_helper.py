"""Small helper classes for testing."""

import typing as tp

import plumbum as pb
from benchbuild.source import Variant, FetchableSource


class BBTestSource(FetchableSource):
    """Source test fixture class."""

    test_versions: tp.List[str]

    def __init__(
        self, test_versions: tp.List[str], local: str,
        remote: tp.Union[str, tp.Dict[str, str]]
    ):
        super().__init__(local, remote)
        self.test_versions = test_versions

    @property
    def local(self) -> str:
        return "test_source"

    @property
    def remote(self) -> tp.Union[str, tp.Dict[str, str]]:
        return "test_remote"

    @property
    def default(self) -> Variant:
        return Variant(owner=self, version=self.test_versions[0])

    # pylint: disable=unused-argument,no-self-use
    def version(self, target_dir: str, version: str) -> pb.LocalPath:
        return pb.local.path('.') / f'varats-test-{version}'

    def versions(self) -> tp.Iterable[Variant]:
        return [Variant(self, v) for v in self.test_versions]
