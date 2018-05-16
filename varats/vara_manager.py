#!/usr/bin/env python3
"""
This module handels the status of VaRA.
Setting up the tooling, keeping it up to date,
and providing necessary information.
"""

import os
import re
import subprocess as sp

from enum import Enum

from plumbum import local, FG
from plumbum.cmd import git, mkdir, ln, ninja, grep, cmake
from plumbum.commands.processes import ProcessExecutionError


def run_with_output(pb_cmd, post_out=lambda x: None):
    """
    Run plumbum command and post output lines to function.
    """
    try:
        with pb_cmd.bgrun(universal_newlines=True,
                          stdout=sp.PIPE, stderr=sp.STDOUT) as p_gc:
            while p_gc.poll() is None:
                for line in p_gc.stdout:
                    post_out(line)
    except ProcessExecutionError:
        post_out("ProcessExecutionError")


def download_repo(dl_folder, url: str, repo_name=None,
                  post_out=lambda x: None):
    """
    Download a repo into the specified folder.
    """
    if not os.path.isdir(dl_folder):
        # TODO: error
        return

    with local.cwd(dl_folder):
        if repo_name is not None:
            git_clone = git["clone", "--progress", url, repo_name]
            run_with_output(git_clone, post_out)
        else:
            git_clone = git["clone", "--progress", url]
            run_with_output(git_clone, post_out)


def add_remote(repo_folder, remote, url):
    """
    Adds new remote to the repository.
    """
    with local.cwd(repo_folder):
        git["remote", "add", remote, url] & FG
        git["fetch", remote] & FG


def fetch_remote(remote, repo_folder=""):
    """
    Fetches the new changes from the remote.
    """
    if repo_folder == '':
        git["fetch", remote] & FG
    else:
        with local.cwd(repo_folder):
            git["fetch", remote] & FG


def checkout_branch(repo_folder, branch):
    """
    Checks out a branch in the repository.
    """
    with local.cwd(repo_folder):
        git["checkout", branch] & FG


def checkout_new_branch(repo_folder, branch, remote_branch):
    """
    Checks out a new branch in the repository.
    """
    with local.cwd(repo_folder):
        git["checkout", "-b", branch, remote_branch] & FG


def get_download_steps():
    """
    Returns the amount of steps it takes to download VaRA. This can be used to
    track the progress during the long download phase.
    """
    return 6


def download_vara(dl_folder, progress_func=lambda x: None,
                  post_out=lambda x: None):
    """
    Downloads VaRA an all other necessary repos from github.
    """
    progress_func(0)
    download_repo(dl_folder, "https://git.llvm.org/git/llvm.git",
                  post_out=post_out)
    dl_folder += "llvm/"
    add_remote(dl_folder, "upstream", "git@github.com:se-passau/vara-llvm.git")

    progress_func(1)
    download_repo(dl_folder + "tools/", "https://git.llvm.org/git/clang.git",
                  post_out=post_out)
    add_remote(dl_folder + "tools/clang/", "upstream",
               "git@github.com:se-passau/vara-clang.git")

    progress_func(2)
    download_repo(dl_folder + "tools/", "git@github.com:se-passau/VaRA.git",
                  post_out=post_out)

    progress_func(3)
    download_repo(dl_folder + "tools/clang/tools/",
                  "https://git.llvm.org/git/clang-tools-extra.git", "extra",
                  post_out=post_out)

    progress_func(4)
    download_repo(dl_folder + "tools/", "https://git.llvm.org/git/lld.git",
                  post_out=post_out)

    progress_func(5)
    download_repo(dl_folder + "projects/",
                  "https://git.llvm.org/git/compiler-rt.git",
                  post_out=post_out)

    progress_func(6)
    mkdir[dl_folder + "build/"] & FG
    with local.cwd(dl_folder + "build/"):
        ln["-s", dl_folder + "tools/VaRA/utils/vara/builds/", "build_cfg"] & FG


def checkout_vara_version(llvm_folder, version, dev):
    """
    Checks out all related repositories to match the VaRA version number.

    ../llvm/ 60 dev
    """
    version = str(version)
    version_name = ""
    version_name += version
    if dev:
        version_name += "-dev"
    print(version_name)
    checkout_new_branch(llvm_folder, "vara-" + version_name,
                        "upstream/vara-" + version_name)
    checkout_new_branch(llvm_folder + "tools/clang/", "vara-" + version_name,
                        "upstream/vara-" + version_name)
    if dev:
        checkout_branch(llvm_folder + "tools/VaRA/", "vara-dev")

    checkout_branch(llvm_folder + "tools/clang/tools/extra/",
                    "release_" + version)
    checkout_branch(llvm_folder + "tools/lld/", "release_" + version)
    checkout_branch(llvm_folder + "projects/compiler-rt/",
                    "release_" + version)


class BuildType(Enum):
    """
    This enum containts all VaRA prepared Build configurations.
    """
    DBG = 1
    DEV = 2
    OPT = 3
    PGO = 4


def get_cmake_var(var_name):
    """
    Fetch the value of a cmake variable from the current cmake config.
    """
    print(grep(var_name, "CMakeCache.txt"))
    # TODO: find way to get cmake var
    raise NotImplementedError


def set_cmake_var(var_name, value):
    """
    Sets a cmake variable in the current cmake config.
    """
    cmake("-D" + var_name + "=" + value, ".")


def init_vara_build(path_to_llvm, build_type: BuildType,
                    post_out=lambda x: None):
    """
    Initialize a VaRA build config.
    """
    full_path = path_to_llvm + "build/"
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    with local.cwd(full_path):
        if build_type == BuildType.DEV:
            cmake = local["./build_cfg/build-dev.sh"]
            run_with_output(cmake, post_out)


def build_vara(path_to_llvm: str, install_prefix: str, build_type: BuildType,
               post_out=lambda x: None):
    """
    Builds a VaRA configuration
    """
    full_path = path_to_llvm + "build/"
    if build_type == BuildType.DEV:
        full_path += "dev/"
    if not os.path.exists(full_path):
        init_vara_build(path_to_llvm, build_type, post_out)

    with local.cwd(full_path):
        set_cmake_var("CMAKE_INSTALL_PREFIX", install_prefix)
        b_ninja = ninja["install"]
        run_with_output(b_ninja, post_out)

###############################################################################
# Git Handling
###############################################################################


class GitState(Enum):
    """
    Represent the direct state of a branch.
    """
    OK = 1
    BEHIND = 2
    ERROR = 3


class GitStatus(object):
    """
    Represents the current update status of a git repository.
    """

    def __init__(self, state, msg: str = ""):
        self.__state = state
        self.__msg = msg

    @property
    def state(self) -> GitState:
        """
        Current state of the git.
        """
        return self.__state

    @property
    def msg(self):
        """
        Additional msg.
        """
        return self.__msg

    def __str__(self):
        if self.state == GitState.OK:
            return "OK"
        elif self.state == GitState.BEHIND:
            return self.msg
        return "Error"


def get_llvm_status(llvm_folder) -> GitStatus:
    """
    Retrieve the git status of llvm.
    """
    with local.cwd(llvm_folder):
        fetch_remote('upstream')
        git_status = git['status']
        stdout = git_status('-sb')
        for line in stdout.split('\n'):
            if line.startswith('## vara-60-dev'):
                match = re.match(r".*\[(.*)\]", line)
                if match is not None:
                    return GitStatus(GitState.BEHIND, match.group(1))
                return GitStatus(GitState.OK)

    return GitStatus(GitState.ERROR)


def get_clang_status(llvm_folder) -> GitStatus:
    """
    Retrieve the git status of clang.
    """
    with local.cwd(llvm_folder + 'tools/clang'):
        fetch_remote('upstream')
        git_status = git['status']
        stdout = git_status('-sb')
        for line in stdout.split('\n'):
            if line.startswith('## vara-60-dev'):
                match = re.match(r".*\[(.*)\]", line)
                if match is not None:
                    return GitStatus(GitState.BEHIND, match.group(1))
                return GitStatus(GitState.OK)

    return GitStatus(GitState.ERROR)


def get_vara_status(llvm_folder) -> GitStatus:
    """
    Retrieve the git status of VaRA.
    """
    with local.cwd(llvm_folder + 'tools/VaRA'):
        fetch_remote('origin')
        git_status = git['status']
        stdout = git_status('-sb')
        for line in stdout.split('\n'):
            if line.startswith('## vara-dev'):
                match = re.match(r".*\[(.*)\]", line)
                if match is not None:
                    return GitStatus(GitState.BEHIND, match.group(1))
                return GitStatus(GitState.OK)

    return GitStatus(GitState.ERROR)


if __name__ == "__main__":
    download_vara("/tmp/foo/")
    checkout_vara_version("/tmp/foo/llvm/", 60, True)
