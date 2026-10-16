import time

from ddmrp_forecast.config.settings import ESCOPO_BUFFER_MESTRE, PASTA_RAW, PASTA_SOURCE, TAMANHO_LOTE
from ddmrp_forecast.io.parquet_store import ler_parquet
from ddmrp_forecast.io.rpt_reader import converter_pasta
from ddmrp_forecast.preprocess.preparacao import preparar_base


def main() -> None:
    # 1) .rpt → Parquet (só converte o que for novo ou alterado)
    converter_pasta(PASTA_SOURCE, PASTA_RAW, TAMANHO_LOTE)

    # 2) leitura do Parquet
    inicio = time.time()
    df = ler_parquet(PASTA_RAW)
    print(f"Registros lidos: {len(df):,} em {time.time() - inicio:,.0f}s "
          f"({df.memory_usage(deep=True).sum() / 1e9:,.2f} GB em memória)")

    # 3) bloco do Colab
    df, removidas = preparar_base(df, ESCOPO_BUFFER_MESTRE)
    print(f"Linhas de componentes removidas: {removidas:,}")
    print(f"Registros após remoção dos componentes: {len(df):,}")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()