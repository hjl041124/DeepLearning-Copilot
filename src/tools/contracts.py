"""Common result contract for future Agent tool adapters."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ToolStatus = Literal["success", "failed"]


@dataclass
class ToolResult:
    """Structured evidence returned by an Agent tool adapter."""

    tool_name: str
    status: ToolStatus
    features: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")

        if self.status not in {"success", "failed"}:
            raise ValueError("status must be 'success' or 'failed'")

        if self.status == "failed" and not self.error:
            raise ValueError("a failed ToolResult must include an error")

        if self.status == "success" and self.error is not None:
            raise ValueError("a successful ToolResult cannot include an error")

    @classmethod
    def success(
        cls,
        tool_name: str,
        *,
        features: dict[str, Any] | None = None,
        flags: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> "ToolResult":
        """Create a successful tool result."""

        return cls(
            tool_name=tool_name,
            status="success",
            features=dict(features or {}),
            flags=dict(flags or {}),
            provenance=dict(provenance or {}),
            warnings=list(warnings or []),
            error=None,
        )

    @classmethod
    def failed(
        cls,
        tool_name: str,
        error: str,
        *,
        features: dict[str, Any] | None = None,
        flags: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> "ToolResult":
        """Create a failed tool result, optionally with partial evidence."""

        return cls(
            tool_name=tool_name,
            status="failed",
            features=dict(features or {}),
            flags=dict(flags or {}),
            provenance=dict(provenance or {}),
            warnings=list(warnings or []),
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary for serializable values."""

        return asdict(self)
