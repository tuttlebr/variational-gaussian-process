#   -*- coding: utf-8 -*-
from pybuilder.core import init, use_plugin

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.flake8")
use_plugin("python.coverage")


name = "variational-gaussian-process"
version = "latest"
default_task = "publish"


@init
def set_properties(project):
    project.set_property("coverage_threshold_warn", 85)
    project.set_property("coverage_break_build", False)
    project.depends_on_requirements("src/main/python/requirements.txt")
    pass
