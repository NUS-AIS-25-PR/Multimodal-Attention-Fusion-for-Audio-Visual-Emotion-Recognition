"""Local filename/completeness audit. Does not claim to decode all media."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from data.ravdess import parse_ravdess_name, _key_from_fields


def expected_keys() -> set[tuple]:
    return {(1, e, intensity, statement, repetition, actor)
            for actor in range(1, 25) for e in range(1, 9)
            for intensity in ([1] if e == 1 else [1, 2])
            for statement in (1, 2) for repetition in (1, 2)}


def audit_dataset(root: Path) -> dict:
    root = Path(root).expanduser().resolve()
    files = {"audio": defaultdict(list), "video": defaultdict(list)}
    expected = expected_keys()
    invalid = []
    manifest = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".wav"}:
            continue
        try:
            fields = parse_ravdess_name(path.name)
        except ValueError:
            # Noise assets unrelated to RAVDESS are ignored, malformed numbered files aren't.
            if path.name[:2].isdigit():
                invalid.append(str(path))
            continue
        if fields["vocal_channel"] != 1:
            continue
        kind = { (2, ".mp4"): "video", (3, ".wav"): "audio" }.get(
            (fields["modality"], path.suffix.lower()))
        if kind is None:
            continue
        key = _key_from_fields(fields)
        files[kind][key].append(str(path))
        stat = path.stat()
        manifest.append([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns])
        if key not in expected:
            invalid.append(str(path))
    audio, video = files["audio"], files["video"]
    paired = set(audio) & set(video)
    unique_pairs = {k for k in paired if len(audio[k]) == len(video[k]) == 1}
    duplicates = [{"modality": kind, "key": list(key), "paths": paths}
                  for kind, mapping in files.items() for key, paths in mapping.items() if len(paths) > 1]
    discovered = sorted({k[-1] for mapping in files.values() for k in mapping})
    actors = {}
    global_counts = Counter(k[1] for k in unique_pairs)
    for actor in sorted(set(range(1, 25)) | set(discovered)):
        counts = Counter(k[1] for k in unique_pairs if k[-1] == actor)
        actors[str(actor)] = {"audio": sum(len(v) for k, v in audio.items() if k[-1] == actor),
                              "video": sum(len(v) for k, v in video.items() if k[-1] == actor),
                              "pairs": sum(counts.values()),
                              "emotions": {str(e): counts[e] for e in range(1, 9)}}
    missing = {"audio": [list(k) for k in sorted(set(video) - set(audio))],
               "video": [list(k) for k in sorted(set(audio) - set(video))]}
    return {"data_root": str(root), "actors": discovered, "by_actor": actors,
            "total_pairs": len(unique_pairs),
            "emotions": {str(e): global_counts[e] for e in range(1, 9)},
            "duplicates": duplicates, "missing_counterparts": missing, "invalid_files": invalid,
            "missing_pair_keys": [list(k) for k in sorted(expected - unique_pairs)],
            "complete": unique_pairs == expected and not duplicates and not invalid
                        and not any(missing.values()),
            "fingerprint": hashlib.sha256(json.dumps(manifest).encode()).hexdigest()}


def require_complete(audit: dict) -> None:
    if not audit["complete"]:
        raise ValueError(f"Incomplete/invalid RAVDESS: {audit['total_pairs']}/1440 unique pairs; "
                         "inspect the audit JSON. Production training refused.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_dataset(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    for actor, counts in report["by_actor"].items():
        print(f"Actor {int(actor):02d}: audio={counts['audio']} video={counts['video']} "
              f"paired={counts['pairs']} emotions={counts['emotions']}")
    print(f"Total paired: {report['total_pairs']}; complete={report['complete']}")
    require_complete(report)


if __name__ == "__main__":
    main()
