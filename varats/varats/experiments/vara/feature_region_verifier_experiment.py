"""
Implements the feature region verifier experiment.

The experiment analyses a project with with feature region analysis, and
compares the results of the dominator to the results of the if-region approach
"""
import typing as tp
from os import times
from pathlib import Path

import benchbuild.utils.actions as actions
from benchbuild import Project
from benchbuild.extensions import compiler, run, time
from benchbuild.utils.cmd import mkdir, opt, timeout

from varats.data.reports.empty_report import EmptyReport
from varats.data.reports.region_verification_report import (
    RegionVerificationReport as FRR,
)
from varats.experiment.experiment_util import (
    exec_func_with_pe_error_handler,
    VersionExperiment,
    get_default_compile_error_wrapped,
    wrap_unlimit_stack_size,
    create_default_compiler_error_handler,
    create_default_analysis_failure_handler,
)
from varats.experiment.wllvm import (
    RunWLLVM,
    get_bc_cache_actions,
    get_cached_bc_file_path,
    BCFileExtensions,
)
from varats.report.report import BaseReport
from varats.report.report import FileStatusExtension as FSE
from varats.report.report import ReportSpecification
from varats.utils.settings import bb_cfg


class FeatureRegionGeneration(actions.Step):  # type: ignore
    """Analyse a project with VaRA and compare dominator and if-region created
    FeatureRegions."""

    NAME = "FeatureRegionGeneration"
    DESCRIPTION = "Analyse the bitcode with -vara-PFTD and -vara-PFTDD -vara-FR-verifier"

    RESULT_FOLDER_TEMPLATE = "{result_dir}/{project_dir}"

    def __init__(self, project: Project, report_spec: ReportSpecification):
        super().__init__(obj=project, action_fn=self.analyze)
        self.__report_spec = report_spec

    def analyze(self) -> actions.StepResult:
        """
        This step performs the actual analysis with the correct command line
        flags.

        Flags used:
        """

        if not self.obj:
            return actions.StepResult.ERROR
        project = self.obj

        # Add to the user-defined path for saving the results of the
        # analysis also the name and the unique id of the project of every
        # run.
        vara_result_folder = self.RESULT_FOLDER_TEMPLATE.format(
            result_dir=str(bb_cfg()["varats"]["outfile"]),
            project_dir=str(project.name)
        )

        mkdir("-p", vara_result_folder)

        timeout_duration = '1h'

        for binary in project.binaries:
            report_type = self.__report_spec.get_report_type("FRR")

            result_file = report_type.get_file_name(
                project_name=str(project.name),
                binary_name=binary.name,
                project_version=project.version_of_primary,
                project_uuid=str(project.run_uuid),
                extension_type=FSE.SUCCESS,
            )

            opt_params = [
                "-vara-PTFD", "-vara-FR-verifier", "-o", "/dev/null",
                get_cached_bc_file_path(
                    project, binary, [
                        BCFileExtensions.NO_OPT, BCFileExtensions.TBAA,
                        BCFileExtensions.BLAME
                    ]
                )
            ]

            run_cmd = opt[opt_params]

            run_cmd = wrap_unlimit_stack_size(run_cmd)

            run_cmd = run_cmd > f"{vara_result_folder}/{result_file}"

            exec_func_with_pe_error_handler(
                run_cmd,
                create_default_analysis_failure_handler(
                    project, report_type, Path(vara_result_folder)
                )
            )


class FeatureRegionVerificationExperiment(VersionExperiment):
    """Generates a commit flow report (CFR) of the project(s) specified in the
    call."""

    NAME = "GenerateFeatureRegionReport"

    REPORT_SPEC = ReportSpecification(FRR)
    REQUIRED_EXTENSIONS = [
        BCFileExtensions.NO_OPT, BCFileExtensions.TBAA, BCFileExtensions.FEATURE
    ]

    def actions_for_project(self, project: Project) -> tp.List[actions.Step]:
        """
        Returns the specified steps to run the project(s) specified in the call
        in a fixed order.

        Args:
            project: to analyze
        """
        # Try, to build the project without optimizations to get more precise
        # blame annotations. Note: this does not guarantee that a project is
        # build without optimizations because the used build tool/script can
        # still add optimizations flags after the experiment specified cflags.
        project.cflags += ["-O1", "-Xclang", "-disable-llvm-optzns", "-g0"]
        # TODO: missing arg for feature model

        project.runtime_extension = run.RuntimeExtension(project, self) \
            << time.RunWithTime()

        # Add the required compiler extensions to the project(s).
        project.compiler_extension = compiler.RunCompiler(project, self) \
            << RunWLLVM() \
            << run.WithTimeout()

        project.compile = get_default_compile_error_wrapped(
            project, self.REPORT_SPEC.main_report,
            FeatureRegionGeneration.RESULT_FOLDER_TEMPLATE
        )

        analysis_actions = get_bc_cache_actions(
            project, self.REQUIRED_EXTENSIONS,
            create_default_compiler_error_handler(
                project, self.REPORT_SPEC.main_report
            )
        )

        analysis_actions.append(
            FeatureRegionGeneration(project, self.REPORT_SPEC)
        )
        analysis_actions.append(actions.Clean(project))

        return analysis_actions
