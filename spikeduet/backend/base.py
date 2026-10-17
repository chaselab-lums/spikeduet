"""Backend interface: ``ExecutionModule -> GeneratedArtifact``."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from spikeduet.ir.execution import ExecutionModule


@dataclass(frozen=True)
class GeneratedArtifact:
    """Generated files as ``{relative_path: text}``; nothing touches the disk until ``write()``."""

    files: Dict[str, str] = field(default_factory=dict)

    def write(self, output_dir) -> None:
        """Writes every file under ``output_dir``, creating it if needed."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, content in self.files.items():
            (out / name).write_text(content)


class Backend:
    """Base class for code generators."""

    name: str = "Backend"

    def generate(self, module: ExecutionModule, **kwargs) -> GeneratedArtifact:
        raise NotImplementedError
