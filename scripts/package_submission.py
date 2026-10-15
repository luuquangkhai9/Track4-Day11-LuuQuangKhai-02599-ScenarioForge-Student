"""Package the requested submission files, including ignored evidence, after the audit passes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_submission.py")], cwd=ROOT, check=True)
    files = [ROOT / name for name in ("REPORT.md", "hazard.json", "requirements.txt", "policy_candidates.json")]
    files += list((ROOT / "lab11").glob("*.py"))
    files += list((ROOT / "scripts").glob("*.py"))
    files += [p for p in (ROOT / "evidence").rglob("*")
              if p.is_file() and p.name != "submission_manifest.json"]
    files = sorted(set(files))
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest_path = ROOT / "evidence/submission_manifest.json"
    manifest_path.write_text(json.dumps(dict(mssv="2202602599", sha256=manifest), indent=2), encoding="utf-8")
    target = ROOT / "submission-2202602599.zip"
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as bundle:
        for path in files + [manifest_path]:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    with ZipFile(target) as bundle:
        assert bundle.testzip() is None
        for name, digest in manifest.items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == digest, name
    print(f"Created {target} ({target.stat().st_size:,} bytes, {len(files)+1} files)")
    print("Submission bundle: use with the original lab checkout or the grader's clean simulator/viewer.")


if __name__ == "__main__":
    main()
