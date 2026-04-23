# Asset source record

**Status: pinned to upstream tagged release `cd-v1.7.1` (Decision D2 closed 2026-04-23).**

Upstream `threehouse-plus-ec/cd-rules` cut `cd-v1.7.0` (at commit `8671c933`) and `cd-v1.7.1` (at commit `ee01c803`) on 2026-04-23, executing the "Tag repo" step of the upstream §15.4 deprecation protocol. Both tags are annotated and retroactive to the Version History entries in `blueprint-threehouse-CD.md` §16. This record now pins to `cd-v1.7.1` — the six asset files referenced below are byte-identical between 1.7.0 and 1.7.1, so the checksum table carries over unchanged from the provisional pin; only `Source commit`, `Source tag`, and the per-file raw URLs change.

The CI hash-drift check (`.github/workflows/ci.yml`, job `cd-local-integrity`) is now a permanent gate.

---

## Source

| Field | Value |
|---|---|
| Source repository | https://github.com/threehouse-plus-ec/cd-rules |
| Source tag        | `cd-v1.7.1` |
| Source commit     | `ee01c80352dd8446f189c3159a3d9e347463902c` |
| Source branch     | `main` |
| Copied on         | 2026-04-17 (re-anchored to `cd-v1.7.1` on 2026-04-23) |
| Recorded by       | U. Warring (AG Schätz) |
| Propagation model | Model B (distributed copy + SHA-256 checksum) per CD 0.10 |

## Files and checksums

| File | SHA-256 | Size (bytes) |
|---|---|---|
| `emblem-16.svg` | `7298b11e6231d27d3454a16380c4d511e1c2251c7c168c5f66cafa8bc05fe44b` | 459 |
| `emblem-32.svg` | `e799f0bd9ddc5d36c6904c43690c8507215ea78d5d12cc5fec53992bade9ff0b` | 464 |
| `emblem-64.svg` | `48bf761c7f903d09b7b98a5cc8b813c035d8c34b31bdece685f0b9c51387fe55` | 458 |
| `tokens.css` | `097b5903dc3983d3215fb46b4b76948a716e5d2448a1175910b884c25af63962` | 1458 |
| `wordmark-full.svg` | `84defa4091307bd59a30b262ea83d1e244da0d317770dbf56661baaa09add0f2` | 678 |
| `wordmark-silent.svg` | `164e8317cd8058c3eb3c3fab8c3a2aa15e2b008300dbb4046f7a0a77090e73fb` | 555 |

Upstream paths (all at repository root on upstream `main`, raw-URL construction):

| File | Upstream raw URL |
|---|---|
| `emblem-16.svg` | `https://raw.githubusercontent.com/threehouse-plus-ec/cd-rules/ee01c80352dd8446f189c3159a3d9e347463902c/emblem-16.svg` |
| `emblem-32.svg` | `https://raw.githubusercontent.com/threehouse-plus-ec/cd-rules/ee01c80352dd8446f189c3159a3d9e347463902c/emblem-32.svg` |
| `emblem-64.svg` | `https://raw.githubusercontent.com/threehouse-plus-ec/cd-rules/ee01c80352dd8446f189c3159a3d9e347463902c/emblem-64.svg` |
| `tokens.css` | `https://raw.githubusercontent.com/threehouse-plus-ec/cd-rules/ee01c80352dd8446f189c3159a3d9e347463902c/tokens.css` |
| `wordmark-full.svg` | `https://raw.githubusercontent.com/threehouse-plus-ec/cd-rules/ee01c80352dd8446f189c3159a3d9e347463902c/wordmark-full.svg` |
| `wordmark-silent.svg` | `https://raw.githubusercontent.com/threehouse-plus-ec/cd-rules/ee01c80352dd8446f189c3159a3d9e347463902c/wordmark-silent.svg` |

## Verification

Two workflows, depending on what you have locally.

**From a `cd-rules` checkout at tag `cd-v1.7.1` (commit `ee01c803`):**

```sh
cd /path/to/cd-rules-checkout
for f in emblem-16.svg emblem-32.svg emblem-64.svg tokens.css wordmark-full.svg wordmark-silent.svg; do
    shasum -a 256 "$f"
done
```

**From assets already placed in `iontrap-dynamics/assets/` (after running `tools/fetch_assets.sh`):**

```sh
for f in emblem-16.svg emblem-32.svg emblem-64.svg tokens.css wordmark-full.svg wordmark-silent.svg; do
    shasum -a 256 "assets/$f"
done
```

**Automated re-population when advancing the pin:** update the `Source commit` and `Source tag` fields to the new tagged commit, then run `tools/hash_assets.sh`. The script fetches each file from the pinned commit, computes SHA-256 and byte-size, and rewrites this checksum table in place.

## Licence

Assets in this folder are distributed under the MIT licence as declared by the upstream `cd-rules` repository. The accompanying `LICENCE` file in this folder records the MIT terms; this `SOURCE.md` records provenance.

## CI hash-drift check

A CI job (`.github/workflows/ci.yml`, `cd-local-integrity`) recomputes SHA-256 and byte-size for each file in `assets/` and compares against the table above. Drift on either column flags a CI failure.

SHA-256 is the authoritative integrity check: any truncation or substitution changes the hash and is detected. Byte-size is a secondary, diagnostic signal — cheap to compute, human-readable in a diff, and useful for triage when a mismatch fires (same-size mismatch suggests encoding or fine-grained edit; different-size mismatch suggests bulk change or truncation). It is sanity, not security.

**Drift-check anchor:** the check runs against tag `cd-v1.7.1` (commit `ee01c803`). Advancing the pin to a future upstream tag is a single-point edit to the `Source commit` and `Source tag` fields above, followed by `tools/hash_assets.sh` to re-populate the checksum table.

---

**Endorsement Marker:** Local candidate framework. Assets consumed from upstream design authority via Model B. No endorsement of `iontrap-dynamics` by `threehouse-plus-ec/cd-rules` is implied by the presence of these assets; consumption is a downstream choice under the terms of the asset licence.
