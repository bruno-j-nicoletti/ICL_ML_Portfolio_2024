from typing import Any, Dict, Protocol

import os
import sys

__all__ = ["Logger"]


class Saveable(Protocol):
    """Protocol for a saveble thing."""

    def save(self, path: str) -> None:
        """Save self to the given file name."""
        pass


_blockStartStr = "################################################################################"


class Logger:
    """A logger."""
    outputDir: str = ""
    baseName: str = ""

    def __init__(self, outputDir: str = "", baseName: str = "") -> None:
        assert (outputDir and baseName) or not (outputDir and baseName)
        if outputDir:
            self.outputDir = outputDir
            if not os.path.exists(outputDir):
                os.makedirs(outputDir)
            self.baseName = baseName
            self.logFile = open(f"{outputDir}/{baseName}.txt", "wt")
        else:
            self.logFile = sys.stdout  # type: ignore

    def log(self, message: str) -> None:
        """Log out a message."""
        self.logFile.write(message + "\n")

    def startBlock(self, message: str = "") -> None:
        """Start a named block of output."""
        self.log(_blockStartStr)
        if message:
            self.log(message)
        self.logFile.flush()

    def endBlock(self) -> None:
        """End a block of output."""
        pass

    def save(self, saveable: Saveable, suffix: str) -> None:
        """Save the saveable out with the given index"""
        if self.outputDir:
            path = f"{self.outputDir}/{self.baseName}{suffix}"
            saveable.save(path)
            self.log(f"Saved '{path}'")

    def logDict(self, d: Dict[str, Any], message: str | None = None) -> None:
        """Log out a dictionary"""
        if message:
            self.log(message)
        for k, v in d.items():
            self.log(f"  {k} : {v}")

    def close(self):
        self.logFile.close()
