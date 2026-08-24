# Input / Output


```python
import mefikit as mf
import numpy as np
import pyvista as pv

pv.set_plot_theme("dark")
pv.set_jupyter_backend("static")
```


```python
volumes = mf.build_cmesh(
    range(2), np.linspace(0.0, 1.0, 5), np.logspace(0.0, 1.0, 5) / 10.0
)
```

## Memory exports

- Through numpy arrays manipulations:
    - medcoupling
    - meshio
    - pyvista
- Through `string` translation to `Python`:
    - json


```python
volumes.to_mc()
```




    MEDCouplingUMesh C++ instance at 0x15319b10. Name : "mf_UMesh". Not set !




```python
volumes.to_pyvista()
```




<div><style>/* PyVista HTML repr stylesheet.
 * Uses pv- prefix to avoid conflicts with other libraries.
 */

:root {
  --pv-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --pv-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --pv-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --pv-border-color: var(--jp-border-color2, #e0e0e0);
  --pv-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --pv-background-color-row-even: var(--jp-layout-color1, #f5f5f5);
  --pv-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
  --pv-badge-active: #1b5e20;
  --pv-badge-normals: #0d47a1;
  --pv-badge-vectors: #00695c;
  --pv-badge-tcoords: #4527a0;
}

body[data-jp-theme-light="false"] {
  --pv-font-color0: var(--jp-content-font-color0, rgba(255, 255, 255, 1));
  --pv-font-color2: var(--jp-content-font-color2, rgba(255, 255, 255, 0.54));
  --pv-font-color3: var(--jp-content-font-color3, rgba(255, 255, 255, 0.38));
  --pv-border-color: var(--jp-border-color2, #424242);
  --pv-disabled-color: var(--jp-layout-color3, #616161);
  --pv-background-color-row-even: var(--jp-layout-color1, #1a1a1a);
  --pv-background-color-row-odd: var(--jp-layout-color2, #252525);
  --pv-badge-active: #66bb6a;
  --pv-badge-normals: #64b5f6;
  --pv-badge-vectors: #4db6ac;
  --pv-badge-tcoords: #b39ddb;
}

html[theme="dark"],
html[data-theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --pv-font-color0: rgba(255, 255, 255, 1);
  --pv-font-color2: rgba(255, 255, 255, 0.54);
  --pv-font-color3: rgba(255, 255, 255, 0.38);
  --pv-border-color: #424242;
  --pv-disabled-color: #616161;
  --pv-background-color-row-even: #1a1a1a;
  --pv-background-color-row-odd: #252525;
  --pv-badge-active: #66bb6a;
  --pv-badge-normals: #64b5f6;
  --pv-badge-vectors: #4db6ac;
  --pv-badge-tcoords: #b39ddb;
}

/* OS-level dark mode fallback: applies when no explicit data-theme is set */
@media (prefers-color-scheme: dark) {
  html:not([data-theme="light"]) {
    --pv-font-color0: rgba(255, 255, 255, 1);
    --pv-font-color2: rgba(255, 255, 255, 0.54);
    --pv-font-color3: rgba(255, 255, 255, 0.38);
    --pv-border-color: #424242;
    --pv-disabled-color: #616161;
    --pv-background-color-row-even: #1a1a1a;
    --pv-background-color-row-odd: #252525;
    --pv-badge-active: #66bb6a;
    --pv-badge-normals: #64b5f6;
    --pv-badge-vectors: #4db6ac;
    --pv-badge-tcoords: #b39ddb;
  }
}

.pv-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
  line-height: 1.6;
  padding-bottom: 4px;
  font-family: var(--jp-ui-font-family, sans-serif);
  font-size: var(--jp-ui-font-size1, 13px);
  color: var(--pv-font-color0);
}

.pv-text-repr-fallback {
  display: none;
}

/* Header */
.pv-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding-top: 6px;
  padding-bottom: 6px;
  border-bottom: solid 1px var(--pv-border-color);
  margin-bottom: 4px;
}

.pv-header-text {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
  flex: 1;
}

.pv-obj-type {
  font-weight: 600;
  color: var(--pv-font-color0);
}

.pv-header-badge {
  display: inline-block;
  font-size: 0.75em;
  font-weight: 600;
  padding: 2px 7px;
  border-radius: 3px;
  color: var(--pv-font-color2);
  border: 1px solid var(--pv-border-color);
  white-space: nowrap;
}

/* Metadata (always-visible key-value rows) */
.pv-metadata {
  margin: 4px 0 6px 0;
  font-size: 0.92em;
  line-height: 1.5;
}

.pv-meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 1px 14px;
  padding: 1px 0;
}

.pv-meta-row-label {
  color: var(--pv-font-color2);
  font-weight: 500;
  white-space: nowrap;
  min-width: 60px;
}

.pv-meta-entry {
  white-space: nowrap;
}

/* Copy-to-clipboard button */
.pv-copy-btn {
  display: inline-block;
  cursor: pointer;
  opacity: 0.5;
  font-size: 0.85em;
  padding: 0 3px;
  vertical-align: middle;
  transition: opacity 0.15s;
  user-select: none;
  border: none;
  background: none;
  color: var(--pv-font-color3);
}

.pv-copy-btn:hover {
  opacity: 1;
  color: var(--pv-font-color0);
}

.pv-meta-label {
  color: var(--pv-font-color3);
  font-weight: 400;
  padding-right: 2px;
}

/* Sections grid */
.pv-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto auto 1fr 20px 20px;
  margin-block-start: 0;
  margin-block-end: 0;
  list-style: none;
}

.pv-section-item {
  display: contents;
}

/* Hidden checkbox for expand/collapse */
.pv-section-item > input {
  display: block;
  opacity: 0;
  height: 0;
  margin: 0;
}

.pv-section-item > input + label {
  color: var(--pv-disabled-color);
}

.pv-section-item > input:enabled + label {
  cursor: pointer;
  color: var(--pv-font-color2);
}

.pv-section-item > input:enabled + label:hover {
  color: var(--pv-font-color0);
}

/* Section summary (left column label) */
.pv-section-summary {
  grid-column: 1;
  color: var(--pv-font-color2);
  font-weight: 500;
  white-space: nowrap;
}

.pv-section-summary > span {
  display: inline-block;
  padding-left: 0.3em;
}

.pv-section-summary-in:disabled + label {
  color: var(--pv-font-color2);
}

/* Expand/collapse arrows */
.pv-section-summary-in + label:before {
  display: inline-block;
  content: "\25b6";
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.pv-section-summary-in:disabled + label:before {
  color: var(--pv-disabled-color);
}

.pv-section-summary-in:checked + label:before {
  content: "\25bc";
}

.pv-section-summary-in:checked + label > span {
  display: none;
}

.pv-section-summary,
.pv-section-inline-details {
  padding-top: 4px;
}

.pv-section-inline-details {
  grid-column: 2 / -1;
}

.pv-section-details {
  grid-column: 1 / -1;
  margin-top: 4px;
  margin-bottom: 5px;
}

.pv-section-summary-in ~ .pv-section-details {
  display: none;
}

.pv-section-summary-in:checked ~ .pv-section-inline-details {
  display: none;
}

.pv-section-summary-in:checked ~ .pv-section-details {
  display: block;
}

.pv-section-summary-in:checked ~ .pv-section-details:has(.pv-var-list) {
  display: contents;
}

/* Variable (array) list */
.pv-var-list,
.pv-var-item {
  display: contents;
}

.pv-var-item > div,
.pv-var-item label,
.pv-var-item > .pv-var-name span {
  background-color: var(--pv-background-color-row-even);
  border-color: var(--pv-background-color-row-odd);
  margin-bottom: 0;
  padding-top: 2px;
}

.pv-var-list > li:nth-child(odd) > div,
.pv-var-list > li:nth-child(odd) > label,
.pv-var-list > li:nth-child(odd) > .pv-var-name span {
  background-color: var(--pv-background-color-row-odd);
  border-color: var(--pv-background-color-row-even);
}

.pv-var-name {
  grid-column: 1;
}

.pv-var-dims {
  grid-column: 2;
}

.pv-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--pv-font-color2);
}

.pv-var-range {
  grid-column: 4;
  color: var(--pv-font-color3);
  font-size: 0.92em;
}

.pv-var-badges {
  grid-column: 5;
  padding-left: 8px;
}

.pv-var-name,
.pv-var-dims,
.pv-var-dtype,
.pv-var-range {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.pv-var-name:hover,
.pv-var-dims:hover,
.pv-var-dtype:hover,
.pv-var-range:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.pv-var-name span {
  padding-left: 25px !important;
}

.pv-var-name-active span {
  font-weight: 600;
}

/* Badges */
.pv-badge {
  display: inline-block;
  font-size: 0.75em;
  font-weight: 600;
  padding: 1px 5px;
  border-radius: 3px;
  vertical-align: middle;
  line-height: 1.4;
}

.pv-badge-active {
  color: var(--pv-badge-active);
  border: 1px solid var(--pv-badge-active);
}

.pv-badge-normals {
  color: var(--pv-badge-normals);
  border: 1px solid var(--pv-badge-normals);
}

.pv-badge-vectors {
  color: var(--pv-badge-vectors);
  border: 1px solid var(--pv-badge-vectors);
}

.pv-badge-tcoords {
  color: var(--pv-badge-tcoords);
  border: 1px solid var(--pv-badge-tcoords);
}

/* Logo and Icons */
.pv-logo {
  display: flex;
  align-items: center;
  flex-shrink: 0;
}

.pv-logo svg {
  width: 28px;
  height: 28px;
}

.pv-brand-logo {
  display: flex;
  align-items: center;
  flex-shrink: 0;
}

.pv-brand-logo svg {
  height: 20px;
  width: auto;
}

/* Children list (MultiBlock / PartitionedDataSet) */
.pv-children-list {
  padding-left: 25px !important;
  list-style: none;
}

.pv-children-list li {
  padding: 1px 0;
}

.pv-child-name {
  font-weight: 500;
}

.pv-child-type {
  color: var(--pv-font-color2);
  font-style: italic;
}

.pv-child-type:before {
  content: "\00b7";
  padding: 0 6px;
  font-style: normal;
}

.pv-child-detail {
  color: var(--pv-font-color3);
  font-size: 0.9em;
}

.pv-child-detail:not(:empty):before {
  content: "\00b7";
  padding: 0 6px;
}
</style><pre class='pv-text-repr-fallback'>UnstructuredGrid (0x7a29c9be5a20)
  N Cells:    16
  N Points:   50
  X Bounds:   0.000e+00, 1.000e+00
  Y Bounds:   0.000e+00, 1.000e+00
  Z Bounds:   1.000e-01, 1.000e+00
  N Arrays:   0</pre><div class='pv-wrap' style='display:none'><div class='pv-header'><span class='pv-logo'><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <defs>
    <linearGradient id="pv-ug-g1" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#ffd040"/>
      <stop offset="100%" stop-color="#376fa0"/>
    </linearGradient>
  </defs>
  <polygon points="3,10 8,5 14,7 13,13 6,14" fill="#376fa0" opacity="0.9"/>
  <polygon points="14,7 22,4 20,12" fill="url(#pv-ug-g1)" opacity="0.85"/>
  <polygon points="22,4 29,9 27,17 20,12" fill="#1a4a70" opacity="0.85"/>
  <polygon points="13,13 20,12 16,20" fill="#ffd040" opacity="0.8"/>
  <polygon points="6,14 13,13 16,20 10,24 4,20" fill="#376fa0" opacity="0.8"/>
  <polygon points="16,20 10,24 18,28" fill="#1a4a70" opacity="0.85"/>
  <polygon points="20,12 27,17 26,25 18,28 16,20" fill="url(#pv-ug-g1)" opacity="0.8"/>
  <g stroke="rgba(255,255,255,0.6)" stroke-width="0.5" fill="none">
    <polygon points="3,10 8,5 14,7 13,13 6,14"/>
    <polygon points="14,7 22,4 20,12"/>
    <polygon points="22,4 29,9 27,17 20,12"/>
    <line x1="13" y1="13" x2="20" y2="12"/>
    <line x1="13" y1="13" x2="16" y2="20"/>
    <line x1="20" y1="12" x2="16" y2="20"/>
    <polygon points="6,14 13,13 16,20 10,24 4,20"/>
    <line x1="16" y1="20" x2="10" y2="24"/>
    <line x1="10" y1="24" x2="18" y2="28"/>
    <line x1="16" y1="20" x2="18" y2="28"/>
    <polygon points="20,12 27,17 26,25 18,28 16,20"/>
  </g>
  <g fill="rgba(255,255,255,0.8)">
    <circle cx="3" cy="10" r="0.9"/>
    <circle cx="8" cy="5" r="0.9"/>
    <circle cx="14" cy="7" r="0.9"/>
    <circle cx="22" cy="4" r="0.9"/>
    <circle cx="29" cy="9" r="0.9"/>
    <circle cx="13" cy="13" r="0.9"/>
    <circle cx="20" cy="12" r="0.9"/>
    <circle cx="6" cy="14" r="0.9"/>
    <circle cx="27" cy="17" r="0.9"/>
    <circle cx="16" cy="20" r="0.9"/>
    <circle cx="4" cy="20" r="0.9"/>
    <circle cx="10" cy="24" r="0.9"/>
    <circle cx="26" cy="25" r="0.9"/>
    <circle cx="18" cy="28" r="0.9"/>
  </g>
</svg>
</span><div class='pv-header-text'><div class='pv-obj-type'>UnstructuredGrid <span class='pv-header-badge'>50 points</span> <span class='pv-header-badge'>16 cells</span> <span class='pv-header-badge'>6 KiB</span></div></div><span class='pv-brand-logo'><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 90 24">
  <text x="0" y="18" font-family="system-ui,-apple-system,sans-serif" font-size="18" font-weight="700" font-style="italic" letter-spacing="-0.5">
    <tspan fill="#3776AB" opacity="0.7">P</tspan><tspan fill="#FFD43B" opacity="0.7">y</tspan><tspan fill="#008c9e">Vista</tspan>
  </text>
</svg>
</span></div><div class='pv-metadata'><div class='pv-meta-row pv-copyable'><span class='pv-meta-row-label'>Bounds</span><button class='pv-copy-btn' title='Copy to clipboard' data-copy='(0.0, 1.0, 0.0, 1.0, 0.1, 1.0)' onclick="navigator.clipboard.writeText(this.dataset.copy)">⧉</button><span class='pv-meta-entry'><span class='pv-meta-label'>X</span> [0.000e+00, 1.000e+00]</span><span class='pv-meta-entry'><span class='pv-meta-label'>Y</span> [0.000e+00, 1.000e+00]</span><span class='pv-meta-entry'><span class='pv-meta-label'>Z</span> [1.000e-01, 1.000e+00]</span></div></div><ul class='pv-sections'></ul></div></div>




```python
volumes.to_meshio()
```




    <meshio mesh object>
      Number of points: 50
      Number of cells:
        hexahedron: 16




```python
volumes.to_json()
```




    '{"coords":{"v":1,"dim":[50,3],"data":[0.0,0.0,0.1,1.0,0.0,0.1,0.0,0.25,0.1,1.0,0.25,0.1,0.0,0.5,0.1,1.0,0.5,0.1,0.0,0.75,0.1,1.0,0.75,0.1,0.0,1.0,0.1,1.0,1.0,0.1,0.0,0.0,0.17782794100389226,1.0,0.0,0.17782794100389226,0.0,0.25,0.17782794100389226,1.0,0.25,0.17782794100389226,0.0,0.5,0.17782794100389226,1.0,0.5,0.17782794100389226,0.0,0.75,0.17782794100389226,1.0,0.75,0.17782794100389226,0.0,1.0,0.17782794100389226,1.0,1.0,0.17782794100389226,0.0,0.0,0.31622776601683794,1.0,0.0,0.31622776601683794,0.0,0.25,0.31622776601683794,1.0,0.25,0.31622776601683794,0.0,0.5,0.31622776601683794,1.0,0.5,0.31622776601683794,0.0,0.75,0.31622776601683794,1.0,0.75,0.31622776601683794,0.0,1.0,0.31622776601683794,1.0,1.0,0.31622776601683794,0.0,0.0,0.5623413251903491,1.0,0.0,0.5623413251903491,0.0,0.25,0.5623413251903491,1.0,0.25,0.5623413251903491,0.0,0.5,0.5623413251903491,1.0,0.5,0.5623413251903491,0.0,0.75,0.5623413251903491,1.0,0.75,0.5623413251903491,0.0,1.0,0.5623413251903491,1.0,1.0,0.5623413251903491,0.0,0.0,1.0,1.0,0.0,1.0,0.0,0.25,1.0,1.0,0.25,1.0,0.0,0.5,1.0,1.0,0.5,1.0,0.0,0.75,1.0,1.0,0.75,1.0,0.0,1.0,1.0,1.0,1.0,1.0]},"element_blocks":{"HEX8":{"cell_type":"HEX8","connectivity":{"Regular":{"v":1,"dim":[16,8],"data":[0,1,3,2,10,11,13,12,2,3,5,4,12,13,15,14,4,5,7,6,14,15,17,16,6,7,9,8,16,17,19,18,10,11,13,12,20,21,23,22,12,13,15,14,22,23,25,24,14,15,17,16,24,25,27,26,16,17,19,18,26,27,29,28,20,21,23,22,30,31,33,32,22,23,25,24,32,33,35,34,24,25,27,26,34,35,37,36,26,27,29,28,36,37,39,38,30,31,33,32,40,41,43,42,32,33,35,34,42,43,45,44,34,35,37,36,44,45,47,46,36,37,39,38,46,47,49,48]}},"fields":{},"families":{"v":1,"dim":[16],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},"groups":{}}}}'



## File read/write

- On rust side, file I/O with the `read`/`write` methods, driven by the file extension:
    - vtk (legacy binary vtk 2.0)
    - yaml
    - json
    - vtkhdf / h5 / hdf5 (HDF5-based VTK)

The legacy vtk reader/writer only supports the old binary vtk 2.0 file format (no rust crate is doing better so far). The HDF5-based `.vtkhdf` reader/writer is the recommended option for a more modern and HPC friendly format. CGNS support is planned.


```python
import pathlib

pathlib.Path("data").mkdir(exist_ok=True)
for ext in ("vtk", "yaml", "json", "vtkhdf"):
    volumes.write(f"data/volumes.{ext}")
    volumes_from_disk = mf.UMesh.read(f"data/volumes.{ext}")
    assert volumes_from_disk
    assert (
        volumes != volumes_from_disk
    )  # this is a new instance, with a different memory adress
```
