from __future__ import annotations
import textwrap
import re
import json
import difflib
from pathlib import Path
from typing import Any


class KB:
    def __init__(
        self, src_dir: str | Path | None = None, db_path: str | Path | None = None
    ):
        self.db: dict[str, dict[str, Any]] = {}
        self.source_dir = Path(src_dir) if src_dir is not None else None
        self.db_path = Path(db_path) if db_path is not None else None
        if self.db_path and self.db_path.exists():
            self.db = json.loads(self.db_path.read_text())
        if self.source_dir:
            for path in self.source_dir.rglob("*"):
                if any((part.startswith(".") for part in path.parts)):
                    continue
                if not path.is_file():
                    continue
                if self.db_path and path == self.db_path:
                    continue
                rel = str(path.relative_to(self.source_dir))
                if rel in self.db:
                    continue
                self.db[rel] = {
                    "id": rel,
                    "parent": None,
                    "children": [],
                    "content": path.read_text(errors="ignore"),
                }
        self.save()

    def __call__(
        self,
        content: str,
        parent: str | None = None,
        id: str | None = None,
        id_prefix: str | None = None,
    ) -> str:
        id = id or self._next_id(id_prefix)
        if parent is not None:
            if parent not in self.db:
                raise KeyError(f"parent does not exist: {parent}")
            self.db[parent]["children"].append(id)
        self.db[id] = {"id": id, "parent": parent, "children": [], "content": content}
        self.save()
        return id

    def __repr__(self):
        return json.dumps(self.db, ensure_ascii=False, indent=2)

    def __getitem__(self, id: str) -> str:
        return self.db[id]["content"]

    def get_branch(
        self, id: str, overrides: dict[str, str] | None = None, indent: bool = True
    ) -> str:

        def branch_format(node: dict[str, Any]) -> str:
            content = (overrides or {}).get(node["id"], node["content"]).strip()
            children = "".join((branch_format(child) for child in node["children"]))
            inner_body = content + (
                children if not children else "\n" + children.strip()
            )
            indented_body = textwrap.indent(inner_body, "  " if indent else "")
            return f'<branch id="{node["id"]}" parent="{node["parent"]}">\n{indented_body}\n</branch>\n'

        tree_data = self.down(id)
        return branch_format(tree_data).strip()

    def get_discussion(self, id: str) -> str:
        chain = []
        curr = self.up(id)
        while curr:
            chain.append(curr)
            curr = curr["parent"]
        chain.reverse()
        xml_elements = []
        for node in chain:
            parent_id = node["parent"]["id"] if node["parent"] else "None"
            xml_elements.append(
                f'<comment id="{node["id"]}" parent="{parent_id}">\n{node["content"].strip()}\n</comment>'
            )
        return "\n".join(xml_elements)

    def get_revision(self, id: str, raw: bool = True) -> str:
        chain = []
        curr = self.up(id)
        while curr:
            chain.append(curr)
            curr = curr["parent"]
        chain.reverse()
        rev_map = {n["id"]: f"v{i}" for i, n in enumerate(chain)}

        def attr(node_id: str) -> str:
            return f'id="{node_id}"' if raw else f'rev="{rev_map[node_id]}"'

        last = chain[-1]
        doc = f"<document {attr(last['id'])}>\n{last['content'].strip()}\n</document>"
        if len(chain) == 1:
            return doc
        diff_lines = []
        for i in range(len(chain) - 1):
            a, b = (chain[i], chain[i + 1])
            a_lines = a["content"].splitlines(keepends=True)
            b_lines = b["content"].splitlines(keepends=True)
            if i == len(chain) - 2:
                diff_lines += list(
                    difflib.unified_diff(
                        a_lines,
                        b_lines,
                        fromfile=rev_map[a["id"]] if not raw else a["id"],
                        tofile=rev_map[b["id"]] if not raw else b["id"],
                        lineterm="",
                        n=0,
                    )
                )
            else:
                raw_diff = list(difflib.unified_diff(a_lines, b_lines, lineterm=""))
                added = sum(
                    (
                        1
                        for l in raw_diff
                        if l.startswith("+") and (not l.startswith("+++"))
                    )
                )
                removed = sum(
                    (
                        1
                        for l in raw_diff
                        if l.startswith("-") and (not l.startswith("---"))
                    )
                )
                a_label = a["id"] if raw else rev_map[a["id"]]
                b_label = b["id"] if raw else rev_map[b["id"]]
                diff_lines.append(f"{a_label} -> {b_label}: +{added} -{removed}")
        diff_block = (
            f"<diff {attr(last['id'])}>\n" + "\n".join(diff_lines) + "\n</diff>"
        )
        return f"{doc}\n{diff_block}"

    def save(self) -> None:
        if self.db_path:
            self.db_path.write_text(repr(self))

    def up(self, id: str) -> dict[str, Any] | None:
        if id is None:
            return None
        node = self.db[id]
        return {
            "id": node["id"],
            "content": node["content"],
            "parent": self.up(node["parent"]),
        }

    def down(self, id: str) -> dict[str, Any]:
        node = self.db[id]
        return {
            "id": node["id"],
            "content": node["content"],
            "parent": node["parent"],
            "children": [self.down(child_id) for child_id in node["children"]],
        }

    def _next_id(self, id_prefix=None) -> str:
        _id_prefix = "c" if id_prefix is None else str(id_prefix)
        i = 1
        while True:
            id = f"{_id_prefix}{i}"
            if id not in self.db:
                return id
            i += 1

    __contains__ = lambda self, x: x in self.db
    __len__ = lambda self: len(self.db)
    __iter__ = lambda self: iter(self.db)


class DocThread:
    def __init__(self, kb: KB | None = None):
        self.kb = KB("mcb.json") if kb is None else kb
        self.submissions = []

    def submit(
        self, content: str, parent: str | None = None, inplace: bool = True
    ) -> str:
        parent_doc_id = None
        root_id = None
        if parent is not None:
            root_id = self._thread_root(parent)
            parent_doc_id = self._parse_doc_id(root_id)
        doc_id = self.kb(content, parent=parent_doc_id, id_prefix="artifact:")
        ref_tg = f'<document src="{doc_id}" />'
        if inplace and root_id:
            self.kb.db[root_id]["content"] = ref_tg
            sub_id = root_id
        else:
            sub_id = self.kb(ref_tg)
            self.submissions.append(sub_id)
        return sub_id

    def comment(self, content: str, to: str) -> str:
        return self.kb(content, parent=to)

    def read(self, comment_id: str) -> str:
        root_id = self._thread_root(comment_id)
        doc_id = self._parse_doc_id(root_id)
        return self.kb.get_branch(
            comment_id, overrides={root_id: self.kb.get_revision(doc_id, raw=False)}
        )

    def _parse_doc_id(self, root_id: str) -> str:
        content = self.kb[root_id]
        m = re.search('<document src="([^"]+)" />', content)
        if not m:
            raise ValueError(f"root node {root_id} has no document src")
        return m.group(1)

    def _thread_root(self, comment_id: str) -> str:
        node = self.kb.db[comment_id]
        while node["parent"] is not None:
            node = self.kb.db[node["parent"]]
        return node["id"]


if __name__ == "__main__":
    kb = KB("src")
    ids = []
    cur_id = "svd.md"
    for i in range(6):
        cur_id = kb(f"cool {i}", cur_id)
        ids.append(cur_id)
    branch_id = ids[3]
    cur_id = branch_id
    for i in range(6):
        cur_id = kb(f"wtf {i}", cur_id)
        ids.append(cur_id)
    print(kb.get_branch(ids[2], indent=True))
