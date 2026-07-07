# PCT6 Support

This repo includes beta support for Helix / Audiotec Fischer `.pct6` tune files used by DSP PC-Tool 6.

What that means:

- `pct6.py` can decode and re-encode **no-password** `.pct6` files
- the decoded inner tune data uses the same `<ATF ...><OC ...><Fil .../></OC>...</ATF>` structure as `.afpx`
- `afpx.py` can inspect that decoded tune structure once you have it in text form

Important limits:

- this is **less proven than `.afpx`**
- it was only verified here against PC-Tool `6.01.08` and `6.03.04`
- password-protected `.pct6` saves are **not supported**
- do **not** decode with UTF-8 replacement if you plan to write the file back

Why the write path is careful:

- some real `.pct6` files contain binary-ish attribute data
- UTF-8 with replacement can silently corrupt untouched metadata
- `pct6.py` uses a byte-preserving latin-1 text view for edit/write workflows

Quick checks:

```powershell
py -3 .\pct6.py selftest
py -3 .\pct6.py decode .\baseline.pct6
```

Real-file safety check:

```python
import pct6
orig = pct6.decode_bytes('baseline.pct6')
pct6.encode(pct6.decode('baseline.pct6'), 'roundtrip_check.pct6')
assert pct6.decode_bytes('roundtrip_check.pct6') == orig
```

Do not hardcode channel order assumptions for `.pct6`.

- PC-Tool 6 files can contain more channels than `.afpx`
- decoded `<OC>` order does not always match the visible Output A/B/C tabs
- confirm channel mapping from the actual tune or PC-Tool screenshots before writing changes
