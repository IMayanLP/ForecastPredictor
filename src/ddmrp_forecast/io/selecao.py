"""Seleção de arquivos pela lista branca de clientes."""

from pathlib import Path


def filtrar_por_cliente(arquivos: list[Path], clientes: tuple[str, ...] | str | None,
                        raiz: str | Path | None = None) -> list[Path]:
    """Mantém os arquivos cujo caminho (pastas a partir de `raiz` + nome) contém algum cliente. None = todos."""
    if clientes is None:
        return arquivos
    if isinstance(clientes, str):   # ("kasvi") sem vírgula é str, não tupla
        clientes = (clientes,)
    clientes = [c.lower() for c in clientes]
    # data/source/<Cliente>_Consumos_.../arquivo: o cliente pode estar só na pasta
    caminhos = [(a.relative_to(raiz).with_suffix("") if raiz is not None else Path(a.stem)).as_posix().lower()
                for a in arquivos]
    return [a for a, c in zip(arquivos, caminhos) if any(cliente in c for cliente in clientes)]
