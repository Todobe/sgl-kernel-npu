#!/usr/bin/env python
# coding=utf-8

"""python api for sgl_kernel_npu."""

import os
import shutil
from configparser import ConfigParser
from pathlib import Path

import setuptools
from setuptools import find_namespace_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from torch_npu.utils.cpp_extension import NpuExtension


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self):
        return True


class Build(build_ext, object):

    def run(self):
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, "build"))
        self.build_temp = os.path.relpath(os.path.join(BASE_DIR, "build/temp"))
        self.library_dirs.append(os.path.relpath(os.path.join(BASE_DIR, "build/lib")))
        super(Build, self).run()


class PackageCANN(build_py):
    def run(self):
        artifacts = Path(os.environ["SGL_CANN_ARTIFACTS"]).resolve()
        for relative in (
            "lib/libsgl_kernel_npu.so",
            "lib/libsgl_kernel_npu_adapter.so",
            "vendors/sgl_kernel_npu/op_api/lib/libcust_opapi.so",
        ):
            if not (artifacts / relative).is_file():
                raise RuntimeError(f"Missing prebuilt artifact: {artifacts / relative}")
        super().run()
        destination = Path(self.build_lib) / "sgl_kernel_npu"
        for directory in ("lib", "vendors"):
            shutil.copytree(artifacts / directory, destination / directory,
                            dirs_exist_ok=True)
        (destination / "_build_features.py").write_text(
            "CANN_OPS = True\n", encoding="utf-8"
        )


WORKING_DIR = Path(__file__).resolve().parent
config = ConfigParser()
config.read(WORKING_DIR / "sgl_kernel_npu" / "config.ini")
if not config.has_section("global"):
    config.read(WORKING_DIR.parents[1] / "config.ini")
_version = config.get("global", "version")


setuptools.setup(
    name="sgl_kernel_npu",
    version=_version,
    description="python api for sgl_kernel_npu",
    packages=find_namespace_packages(
        include=("sgl_kernel_npu", "sgl_kernel_npu.*"),
        exclude=("sgl_kernel_npu.lib*", "sgl_kernel_npu.vendors*"),
    ),
    ext_modules=[] if os.environ.get("SGL_CANN_ARTIFACTS") else [
        NpuExtension("sgl_kernel_npu._C", sources=[])
    ],
    distclass=BinaryDistribution,
    cmdclass={"build_py": PackageCANN} if os.environ.get("SGL_CANN_ARTIFACTS") else {},
    url="https://github.com/sgl-project/sgl-kernel-npu/",
    license="BSD 3 License",
    python_requires=">=3.7",
    package_data={"sgl_kernel_npu": ["lib/**", "VERSION"]},
)
