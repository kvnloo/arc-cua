"""Apply the narrow #5 fix to the exact reviewed upstream source, or fail closed."""
from hashlib import sha1
from pathlib import Path

BASE_BLOBS = {
    "src/arc_cua/safety.py": "1e1c3061134e13cdd3adcaec5249ef688a880b8a",
    "src/arc_cua/runtime.py": "b500099858cfbff6f712018bedd2a6c7c0b11a99",
    "src/arc_cua/policies/choice.py": "667368ab9c4af5f19ba04cbf063c2b11a04fb187",
}


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one occurrence: {old!r}")
    return text.replace(old, new, 1)


def main() -> None:
    originals = {}
    for path, expected in BASE_BLOBS.items():
        data = Path(path).read_bytes()
        actual = sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Refusing unexpected source {path}: {actual} != {expected}")
        originals[path] = data.decode()

    safety = originals["src/arc_cua/safety.py"]
    safety = replace_once(safety, "Risky controls are recognized by whole words in their label. A subtask opts into a\ncategory", "Risky controls are recognized by label words and structured window-control metadata.\nA subtask opts into a category")
    safety = replace_once(safety, "from typing import Any, Iterable\n", "from typing import TYPE_CHECKING, Any, Iterable\n\nif TYPE_CHECKING:\n    from .models import DesktopElement\n")
    safety = replace_once(safety, "\ndef redact(value: Any, secrets: Iterable[str]) -> Any:\n", '''
def element_risks(element: DesktopElement) -> set[str]:
    """Combine label risks with backend-supplied control semantics."""
    risks = risks_of(element.name)
    # Native title-bar close buttons can be unlabeled (or localized).
    # Add this risk; do not let metadata erase another risk in the label.
    if element.metadata.get("window_control") == "close":
        risks.add("close")
    return risks


def disallowed_element_risks(element: DesktopElement, allowed: Iterable[str]) -> set[str]:
    return element_risks(element) - set(allowed)


def redact(value: Any, secrets: Iterable[str]) -> Any:
''')
    changed = {"src/arc_cua/safety.py": safety}
    for path, target in (("src/arc_cua/runtime.py", "target"), ("src/arc_cua/policies/choice.py", "element")):
        text = replace_once(originals[path], "disallowed_risks, redact", "disallowed_element_risks, redact")
        text = replace_once(text, f"disallowed_risks({target}.name, subtask.allowed_risks)", f"disallowed_element_risks({target}, subtask.allowed_risks)")
        changed[path] = text
    # All source checks and transformations finish before writing any file.
    for path, text in changed.items():
        Path(path).write_text(text)


if __name__ == "__main__":
    main()
