import os
from glob import glob
from setuptools import find_packages, setup

package_name = "aic_foxglove"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@todo.todo",
    description="AIC Foxglove utilities and visualization markers",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "publish_scene_markers = aic_foxglove.publish_scene_markers:main",
        ],
    },
)
