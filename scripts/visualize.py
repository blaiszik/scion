#!/usr/bin/env python3
"""
Quick local viewer for Scion fold outputs (mmCIF).

Generates a self-contained HTML page with 3DMol.js loaded from a CDN
and opens it in your default browser. No Python dependencies beyond
the standard library — the CIF text is embedded directly, so the
resulting HTML is portable (share, archive, attach).

Usage:
    # Once on the local machine:
    scp polaris.alcf.anl.gov:~/scion-deploy/polaris_demo.cif .
    python scripts/visualize.py polaris_demo.cif

    # Other common forms:
    python scripts/visualize.py path/to/structure.cif -o report.html
    python scripts/visualize.py polaris_demo.cif --no-browser  # CI / headless
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from pathlib import Path


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Scion fold — {title}</title>
<script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.0/build/3Dmol-min.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; background: #1a1a1a; color: #eaeaea;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }}
  header {{ padding: 10px 18px; border-bottom: 1px solid #333; display: flex;
            gap: 18px; align-items: center; }}
  header h1 {{ font-size: 14px; font-weight: 600; margin: 0; }}
  header .meta {{ font-size: 12px; color: #888; }}
  .controls {{ margin-left: auto; display: flex; gap: 6px; }}
  button {{ background: #2a2a2a; color: #eaeaea; border: 1px solid #444;
            padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; }}
  button:hover {{ background: #333; }}
  button.active {{ background: #0053d6; border-color: #0053d6; }}
  #viewer {{ width: 100vw; height: calc(100vh - 47px); position: relative; }}
  .legend {{ position: absolute; bottom: 12px; left: 12px; padding: 8px 12px;
             background: rgba(26, 26, 26, 0.85); border: 1px solid #333;
             border-radius: 4px; font-size: 11px; display: none; }}
  .legend.show {{ display: block; }}
  .legend .row {{ display: flex; align-items: center; gap: 6px; }}
  .legend .swatch {{ width: 14px; height: 14px; border-radius: 2px; display: inline-block; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <span class="meta">{size_kb} KiB</span>
  <div class="controls">
    <button id="btn-spectrum" class="active">N→C spectrum</button>
    <button id="btn-plddt">pLDDT</button>
    <button id="btn-spin">spin</button>
  </div>
</header>
<div id="viewer">
  <div id="legend" class="legend">
    <div class="row"><span class="swatch" style="background:#0053D6"></span>very high (&gt; 90)</div>
    <div class="row"><span class="swatch" style="background:#65CBF3"></span>confident (70–90)</div>
    <div class="row"><span class="swatch" style="background:#FFDB13"></span>low (50–70)</div>
    <div class="row"><span class="swatch" style="background:#FF7D45"></span>very low (&lt; 50)</div>
  </div>
</div>
<script>
  const cif = {cif_json};
  const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "#1a1a1a" }});
  viewer.addModel(cif, "cif");

  // Detect pLDDT B-factor scale (Boltz writes 0-1, AlphaFold writes 0-100).
  // Probe a few atoms; if all observed B-factors are <= 1.0, treat as 0-1.
  function plddtScale() {{
    const atoms = viewer.selectedAtoms({{}});
    let maxB = 0;
    for (let i = 0; i < Math.min(atoms.length, 200); i++) {{
      if (atoms[i].b !== undefined && atoms[i].b > maxB) maxB = atoms[i].b;
    }}
    return maxB > 1.5 ? 1.0 : 100.0;
  }}

  function applySpectrum() {{
    viewer.setStyle({{}}, {{ cartoon: {{ color: "spectrum" }} }});
    viewer.addStyle({{ hetflag: true }}, {{ stick: {{}} }});
    document.getElementById("legend").classList.remove("show");
    viewer.render();
  }}
  function applyPlddt() {{
    const scale = plddtScale();
    viewer.setStyle({{}}, {{
      cartoon: {{
        colorfunc: function(atom) {{
          const b = (atom.b ?? 0) * scale;
          if (b > 90) return "#0053D6";
          if (b > 70) return "#65CBF3";
          if (b > 50) return "#FFDB13";
          return "#FF7D45";
        }}
      }}
    }});
    viewer.addStyle({{ hetflag: true }}, {{ stick: {{}} }});
    document.getElementById("legend").classList.add("show");
    viewer.render();
  }}

  let spinning = false;
  function setActive(id) {{
    for (const b of ["btn-spectrum", "btn-plddt"]) {{
      document.getElementById(b).classList.toggle("active", b === id);
    }}
  }}
  document.getElementById("btn-spectrum").onclick = () => {{ setActive("btn-spectrum"); applySpectrum(); }};
  document.getElementById("btn-plddt").onclick    = () => {{ setActive("btn-plddt");    applyPlddt();    }};
  document.getElementById("btn-spin").onclick = () => {{
    spinning = !spinning;
    viewer.spin(spinning ? "y" : false);
    document.getElementById("btn-spin").classList.toggle("active", spinning);
  }};

  applySpectrum();
  viewer.zoomTo();
  viewer.render();
</script>
</body>
</html>
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a Scion fold output (mmCIF) as a standalone HTML page.",
    )
    parser.add_argument("cif", type=Path, help="Path to an mmCIF file.")
    parser.add_argument(
        "-o", "--output",
        type=Path, default=None,
        help="Output HTML path (default: same as CIF with .html suffix).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Write the HTML but don't open it. Useful in headless / CI runs.",
    )
    args = parser.parse_args(argv)

    if not args.cif.exists():
        print(f"Error: {args.cif} not found.", file=sys.stderr)
        return 1

    cif_text = args.cif.read_text()
    if "_atom_site" not in cif_text:
        print(
            f"Warning: {args.cif} does not look like an mmCIF "
            f"(no `_atom_site` loop found). Rendering anyway.",
            file=sys.stderr,
        )

    out = args.output or args.cif.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        TEMPLATE.format(
            title=html.escape(args.cif.name),
            size_kb=f"{len(cif_text) / 1024:.1f}",
            cif_json=json.dumps(cif_text),
        )
    )
    print(f"Wrote {out}  ({out.stat().st_size} bytes)")

    if not args.no_browser:
        webbrowser.open(f"file://{out.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
