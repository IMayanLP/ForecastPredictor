"""Seleção de arquivos pela lista branca de clientes."""

from pathlib import Path


def filtrar_por_cliente(arquivos: list[Path], clientes: tuple[str, ...] | str | None) -> list[Path]:
    """Mantém os arquivos cujo nome contém algum cliente da lista. None = todos."""
    if clientes is None:
        return arquivos
    if isinstance(clientes, str):   # ("kasvi") sem vírgula é str, não tupla
        clientes = (clientes,)
    clientes = [c.lower() for c in clientes]
    return [a for a in arquivos if any(c in a.stem.lower() for c in clientes)]
