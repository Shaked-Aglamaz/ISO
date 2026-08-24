"""Prepare published-figure panels for the background slides of the defense deck.

Every panel here comes from a paper the thesis already cites, and every slide that
uses one carries the citation. Sources:

  purcell_*      Purcell et al. 2017, Nat Commun 8:15930 (open access, CC BY)
                 downloaded from PMC5490197.
  lecci_*        Lecci et al. 2017, Sci Adv 3:e1602026 (open access)
                 downloaded from PMC5298853.
  champetier_*   Champetier et al. 2023, Sleep 46:zsac282 — rendered from the local
                 PDF in thesis/references/.
  lazar_*        Lazar et al. 2019, J Neurosci Methods 316:22-34 — local PDF.
  nir_lc_*       Nir et al., LC-NE and sleep review (unpublished manuscript in
                 thesis/references/) — ASK YUVAL before showing this one.

The originals live in results/defense_slides_V1/refs/; this script writes the
cropped, slide-ready panels next to them.

Run from repo root with the venv active:
    PYTHONIOENCODING=utf-8 python code/defense/make_ref_figures.py
"""
from __future__ import annotations

from pathlib import Path

import pymupdf
from PIL import Image

REFS = Path("results/defense_slides_V1/refs")
PDFS = Path("thesis/references")


def crop(src, box, out, scale=1.0):
    """box is (left, top, right, bottom) as fractions of width/height."""
    img = Image.open(REFS / src)
    w, h = img.size
    l, t, r, b = box
    cropped = img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
    if scale != 1.0:
        cropped = cropped.resize((int(cropped.width * scale), int(cropped.height * scale)),
                                 Image.LANCZOS)
    cropped.convert("RGB").save(REFS / out)
    print(f"  {out}  {cropped.size}")
    return cropped


def render_pdf_page(pdf_name, page_no, out, dpi=200):
    doc = pymupdf.open(PDFS / pdf_name)
    pix = doc[page_no].get_pixmap(dpi=dpi)
    pix.save(REFS / out)
    print(f"  {out}  {pix.width}x{pix.height}")


def main():
    print("cropping published panels:")

    # Purcell 2017 Fig 6a,b — slow (frontal) vs fast (central-parietal) spindle density.
    # The two panels sit side by side in the paper, which makes a 4:1 strip that shrinks
    # to nothing on a slide; stacked, the same two panels fill a slide box.
    src = Image.open(REFS / "purcell_f6.jpg")
    w, h = src.size
    slow = src.crop((int(0.03 * w), 0, int(0.47 * w), int(0.205 * h)))
    fast = src.crop((int(0.50 * w), 0, int(0.95 * w), int(0.205 * h)))
    scale = 2.4
    slow = slow.resize((int(slow.width * scale), int(slow.height * scale)), Image.LANCZOS)
    fast = fast.resize((int(fast.width * scale), int(fast.height * scale)), Image.LANCZOS)
    stacked = Image.new("RGB", (max(slow.width, fast.width),
                                slow.height + fast.height + 16), "white")
    stacked.paste(slow, (0, 0))
    stacked.paste(fast, (0, slow.height + 16))
    stacked.save(REFS / "purcell_spindle_topo.png")
    print(f"  purcell_spindle_topo.png  {stacked.size}")

    # Purcell 2017 Fig 2b — spindle density by frequency, one curve per decade of age
    crop("purcell_f2.jpg", (0.585, 0.055, 1.0, 0.49), "purcell_spindle_age.png", scale=2.2)

    # Lecci 2017 Fig 1C left (mouse) and 1G left (human): the ~0.02 Hz sigma spectrum
    crop("lecci_f1.jpg", (0.0, 0.197, 0.47, 0.355), "lecci_mouse_spectrum.png", scale=2.0)
    crop("lecci_f1.jpg", (0.0, 0.597, 0.47, 0.752), "lecci_human_spectrum.png", scale=2.0)

    # Champetier 2023 Fig 2 — the ISFS in young-middle aged vs older adults (C3/C4)
    render_pdf_page("Age_changes_spindle_memory_consolidation_Champetier_2023.pdf", 6,
                    "champetier_p7_hi.png", dpi=220)
    crop("champetier_p7_hi.png", (0.07, 0.055, 0.95, 0.375), "champetier_iso_age.png")

    # Lazar 2019 Fig 3, top-left panel only (5-15 mHz infra power of the sigma band by
    # electrode). The electrode labels belong to the shared x-axis and are printed under
    # the bottom row of the figure, so the label strip is stitched under the panel; the
    # x positions are identical, nothing is rescaled and no value is touched.
    render_pdf_page("ISFS_humans_Lazar_2019.pdf", 3, "lazar_p4_hi.png", dpi=220)
    page = Image.open(REFS / "lazar_p4_hi.png")
    w, h = page.size
    panel = page.crop((int(0.06 * w), int(0.600 * h), int(0.393 * w), int(0.760 * h)))
    labels = page.crop((int(0.06 * w), int(0.905 * h), int(0.393 * w), int(0.952 * h)))
    out = Image.new("RGB", (panel.width, panel.height + labels.height), "white")
    out.paste(panel, (0, 0))
    out.paste(labels, (0, panel.height))
    out.save(REFS / "lazar_iso_topography.png")
    print(f"  lazar_iso_topography.png  {out.size}")

    # Nir et al. review, panel B: LC activity, spindle density, micro-arousals and
    # autonomic arousal on the same infra-slow cycle
    crop("nir_review_image3.png", (0.03, 0.38, 1.0, 1.0), "nir_lc_infraslow.png")


if __name__ == "__main__":
    main()
