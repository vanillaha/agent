"""存储层（M5 起）。

M5-5：本地知识库 SQLite 落库；M4 将在此扩展去重与版本关系表。
"""

from paper_evidence_qa.storage.kb_store import KBStore

__all__ = ["KBStore"]
