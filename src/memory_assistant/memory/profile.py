"""用户画像的数据模型与 prompt 渲染。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UserProfile:
    """用户稳定属性的轻量模型，字段名对应事实谓词。"""

    user_id: str
    attributes: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, user_id: str, data: dict | None) -> "UserProfile":
        return cls(user_id=user_id, attributes=dict(data or {}))

    def render(self) -> str:
        """渲染成短小、稳定的文本供系统 prompt 使用。"""
        return "\n".join(
            f"- {key}：{value}"
            for key, value in sorted(self.attributes.items())
            if value is not None and str(value).strip()
        )


__all__ = ["UserProfile"]
