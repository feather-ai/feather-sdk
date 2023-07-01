import pytest
from feather import helpers
import os

def test_PathClean():  
    cwd = os.getcwd()

    assert helpers.CleanRelativePath(cwd, "dirA/fileA.txt") == "dirA/fileA.txt"

    assert helpers.CleanRelativePath(cwd, os.path.join(cwd, "dirA/fileA.txt")) == "dirA/fileA.txt"

    assert helpers.CleanRelativePath(cwd, "dirA\\fileA.txt") == "dirA/fileA.txt"
