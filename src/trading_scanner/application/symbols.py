"""Loading configured market symbols from a text file."""

from pathlib import Path


class SymbolLoadError(FileNotFoundError):
    """Raised when the scanner symbol file cannot be read."""


class SymbolLoader:
    """Load unique, non-empty symbols while retaining their file order."""

    def load(self, path: Path) -> list[str]:
        """Read symbols from *path* and discard blank or duplicate entries."""
        if not path.is_file():
            raise SymbolLoadError(f"Symbols file does not exist: {path}")

        symbols: list[str] = []
        seen: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            symbol = line.strip()
            if symbol and symbol not in seen:
                symbols.append(symbol)
                seen.add(symbol)
        return symbols
