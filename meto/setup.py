'''
-----------------------------------------------------------------------------
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import os
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

_src_path = os.path.dirname(os.path.abspath(__file__))

setup(
    name="meto",
    version="0.0.1",
    description="a mesh tokenization library",
    ext_modules=[
        Pybind11Extension(
            name="_meto",
            sources=["src/bindings.cpp"], # just cpp files
            include_dirs=[os.path.join(_src_path, "include")],
            extra_compile_args=["-std=c++17", "-O3"],
        ),
    ],
    cmdclass={"build_ext": build_ext},
    install_requires=["numpy", "pybind11", "trimesh", "kiui", "pymeshlab"],
)