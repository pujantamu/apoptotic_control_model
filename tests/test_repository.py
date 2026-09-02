from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_figure_scripts_are_present():
    main = sorted((ROOT / "scripts" / "main").glob("figure_*.py"))
    supplementary = sorted((ROOT / "scripts" / "si").glob("figure_s*.py"))
    assert len(main) == 4
    assert len(supplementary) == 8


def test_final_figures_are_present():
    main = sorted((ROOT / "figures" / "main").glob("Figure_*.png"))
    supplementary = sorted((ROOT / "figures" / "supplementary").glob("Figure_S*.png"))
    assert len(main) == 4
    assert len(supplementary) == 8
