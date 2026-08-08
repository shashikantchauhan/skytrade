from pathlib import Path

import pytest

from trading_scanner.application.symbols import SymbolLoader, SymbolLoadError


def test_symbol_loader_ignores_blank_lines_and_duplicates(tmp_path: Path) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("RELIANCE.NS\n\n SBIN.NS \nRELIANCE.NS\n", encoding="utf-8")

    assert SymbolLoader().load(symbols_file) == ["RELIANCE.NS", "SBIN.NS"]


def test_symbol_loader_reports_missing_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(SymbolLoadError, match="Symbols file does not exist"):
        SymbolLoader().load(missing_file)
