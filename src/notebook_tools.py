"""Strict, compact scientific presentation shared by the notebook chain."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from pathlib import Path
from html import escape
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
import pandas as pd
from IPython.display import HTML, display

FONT_SIZE = 18
TITLE_SIZE = 20
FIGURE_DPI = 350


def configure_style() -> None:
    """Require genuine Times New Roman and the figure scale."""
    # Users may provide licensed font files locally without bundling them in Git.
    font_directory = Path(__file__).resolve().parents[1] / "fonts"
    if font_directory.is_dir():
        for extension in ("*.ttf", "*.otf", "*.ttc"):
            for font_file in font_directory.glob(extension):
                font_manager.fontManager.addfont(str(font_file))
    try:
        font_manager.findfont("Times New Roman", fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError(
            "Times New Roman is required. Install it locally or place licensed "
            "font files in fonts/; see README.md, Figure typography."
        ) from error
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "legend.title_fontsize": FONT_SIZE,
            "figure.titlesize": TITLE_SIZE,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.formatter.useoffset": False,
            "axes.unicode_minus": True,
        }
    )
    # Prevent the inline backend from changing the specified font or DPI.
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("png")


def display_grouped_tables(
    dataframe: pd.DataFrame,
    identifier_columns: Sequence[str],
    column_groups: Mapping[str, Sequence[str]],
    digits: int = 4,
) -> None:
    """Emit one HTML output containing complete semantically grouped tables."""
    identifiers = list(identifier_columns)
    metrics = [column for columns in column_groups.values() for column in columns]
    assert len(metrics) == len(set(metrics))
    assert set(metrics) == set(dataframe.columns) - set(identifiers)
    assert dataframe[identifiers].notna().all().all()
    sections = []
    for title, columns in column_groups.items():
        compact = dataframe.loc[:, identifiers + list(columns)].copy()
        assert len(compact) == len(dataframe)
        rendered = compact.to_html(
            index=False,
            max_rows=None,
            max_cols=None,
            float_format=lambda value: format_scientific_number(value, digits),
            na_rep="Not applicable",
        )
        assert "..." not in rendered and "…" not in rendered
        sections.append(f"<h4>{escape(title)}</h4>{rendered}")
    display(HTML('<div class="scientific-result">' + "".join(sections) + "</div>"))


def show_table(
    dataframe: pd.DataFrame,
    identifiers: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    path: Path,
    digits: int = 4,
) -> None:
    """Save the exact numerical table silently and show a complete summary."""
    dataframe.to_csv(path, index=False)
    display_grouped_tables(dataframe, identifiers, groups, digits)


def format_scientific_number(value: float, digits: int = 4) -> str:
    """Keep extreme magnitudes legible without hiding their sign or precision."""
    if abs(value) >= 1_000_000 or 0 < abs(value) < 10 ** (-digits):
        return scientific_minus(f"{value:.{digits}e}")
    return scientific_minus(f"{value:.{digits}f}")


def scientific_minus(text: str) -> str:
    """Use U+2212 for signed numbers and subtraction, preserving name hyphens."""
    text = re.sub(r"(?<![A-Za-zА-Яа-я])-(?=\d|\.\d|∞)", "−", text)
    text = re.sub(r"(?<=[eE])-(?=\d)", "−", text)
    return text.replace(" - ", " − ")


def save_figure(fig: plt.Figure, path: str | Path) -> None:
    """Validate typography and export the same canvas as PNG and vector PDF."""
    path = Path(path)
    fig.canvas.draw()
    for artist in fig.findobj(Text):
        artist.set_text(scientific_minus(artist.get_text()))
    fig.canvas.draw()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text().strip():
            assert 18 <= text.get_fontsize() <= 20, (
                text.get_text(),
                text.get_fontsize(),
            )
            assert "Times New Roman" in text.get_fontfamily()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(
        path.with_suffix(".pdf"), dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.15
    )
