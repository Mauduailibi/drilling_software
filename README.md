# Drilling Software

Software desktop (PySide6) com dois módulos independentes:

- **Well Path Correction** (`drilling.features.well_path`) — correção de trajetória 3D (Cases 1, 2 e 3).
- **Minimization** (`drilling.features.minimization`) — otimização de trajetória Tipo 1 (força, torque, tempo de broca, tempo total) sobre uma malha de reservatório pronta, lida de um arquivo GRDECL.

A matemática dos solvers está congelada pelos testes golden. Qualquer mudança de fórmula, limite ou default da GUI deve falhar em `pytest`.

Tipos compartilhados (`Point3D`, `Point2D`, `WellPathInput`, `ConstraintCheck`) e o parse de campos `x, y[, z]` ficam em `drilling.core`. Os dois módulos continuam independentes e usam convenções de eixo diferentes (Z negativo × Z positivo): **não há conversão automática entre eles**.

## Coordenadas (não misturar)

| Módulo | Eixos | Profundidade |
|--------|--------|----------------|
| Well Path | XYZ, metros | **Z negativo** |
| Minimization | XYZ, metros (poço no plano vertical P0 → P3) | **Z positivo** para baixo |

## Malha geológica (GRDECL)

O sistema não constrói malhas: ele lê uma malha corner-point pronta no formato GRDECL (Eclipse), posiciona nela a cabeça do poço e o alvo e calcula a trajetória.

Na aba Minimization:

1. **Load GRDECL...** carrega o arquivo (obrigatório para rodar). A tabela lista as litologias encontradas e o número de células ativas de cada uma.
2. **Wellhead (x, y, z)** e **Target (x, y, z)** usam as coordenadas da malha, com profundidade positiva para baixo (a mesma convenção de `ZCORN`). O poço Tipo 1 fica no plano vertical que passa pelos dois pontos. O alvo precisa cair em uma célula ativa da malha; senão a otimização não começa e a mensagem mostra a extensão da malha.
3. O ROP base de cada elemento é **Base ROP × coeficiente da litologia**. A linha *Outside grid* vale para o trecho do poço fora da malha (ou em células inativas). A mesma tabela tem o fator de desgaste de broca por litologia.
4. A aba **3D Grid** mostra as células das colunas atravessadas pelo poço e as trajetórias ótimas; **Zoom to grid** enquadra só esse trecho.

Do arquivo, só são lidos `SPECGRID`/`DIMENS`, `COORD`, `ZCORN`, `ACTNUM` e a litologia; todo o resto é ignorado. A litologia vem de um keyword inteiro `FACIES`/`LITHOLOGY`/`LITHO` ou, na falta dele, das frações `SED1`, `SED2`, ...: a célula recebe o sedimento de maior fração. Coordenadas são usadas como estão no arquivo (`MAPAXES` é ignorado) e a coluna de cada ponto é localizada supondo pilares aproximadamente verticais.

```python
from drilling.features.minimization import GridGeology, read_grdecl

grid = read_grdecl("tests/data/kvl_quarter_five_spot.grdecl")
geology = GridGeology(grid, wellhead=(-40, 500, -2975), target=(960, 500, 25),
                      base_rop=15.0, rop_coefficients={"SED1": 1.5, "SED3": 0.7})
geology.segment_at(950.0, 2990.0)   # {'lithology': 'SED2', 'rop': 15.0}
```

Malhas de teste:

- `tests/data/kvl_quarter_five_spot.grdecl`: exportação real do KVL, mas é um modelo mínimo (10 × 10 × 3 células de 100 × 100 × 10 m, só 30 m de espessura). Serve para testar o leitor, não para otimizar um poço de 3 km.
- `python scripts/make_test_grid.py` gera `outputs/synthetic_basin.grdecl`: 3 × 3 km, 12 camadas até ~3,5 km, mergulho, anticlinal e um canal arenoso. Os valores padrão de Wellhead/Target da GUI caem dentro dela.

A `mesh` de intervalos de profundidade (`build_default_mesh`) continua existindo só como geologia dos testes golden e dos scripts antigos.

## Ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Subir a GUI (pacote instalável):

```bash
python -m drilling.app
```

Atalhos equivalentes: `python -m drilling` e o script `drilling` instalado pelo pacote.

## Testes golden

Os snapshots em `tests/goldens/` registram os números atuais dos defaults. Tolerância: `1e-10`.

```bash
# Tudo, inclusive os quatro objetivos no grid L1 × R (~30 s)
pytest

# Sem as varreduras completas do grid
pytest -m "not slow"
```

Regenerar snapshots **somente** se a mudança numérica for intencional:

```bash
python scripts/capture_goldens.py        # rápido
python scripts/capture_goldens.py --slow # inclui os 4 ótimos (alguns minutos)
```

## Scripts de pesquisa

Scripts de pesquisa (não são testes) ficam em `scripts/` e usam o mesmo pacote da GUI. É o lugar para testar e visualizar resultados rapidamente, sem abrir a interface. Arquivos gerados vão para `outputs/` (ignorado pelo git).

```bash
python scripts/optimization_demo.py      # quatro objetivos + gráficos
python scripts/selected_trajectory.py    # inspeciona um par (L1, R) escolhido
python scripts/make_test_grid.py         # gera outputs/synthetic_basin.grdecl
python scripts/grid_demo.py outputs/synthetic_basin.grdecl --coef SED1=1.6   # otimização sobre uma GRDECL + figuras 2D/3D
```

## Fluxo de trabalho

Todo o desenvolvimento acontece neste repositório, inclusive features novas da otimização (não há mais uma cópia separada do código).

1. `main` é sempre estável: `pytest` passa e a GUI abre.
2. Cada mudança nasce em uma branch a partir de `main` atualizada:
   - `feature/<nome>` para funcionalidades (ex.: `feature/geology-editor`);
   - `fix/<nome>` para correções;
   - `docs/<nome>` para documentação.
3. Commits em inglês, no imperativo, com uma frase de título terminada em ponto e um corpo explicando o porquê.
4. Para testar e visualizar, use ou crie um script em `scripts/` (saídas em `outputs/`). Se o resultado virar referência, transforme-o em teste em `tests/`.
5. Mudança numérica intencional: regenere os goldens com `scripts/capture_goldens.py` em um commit próprio, explicando o que mudou e por quê.
6. Abra um Pull Request para `main`, com `pytest` passando; outra pessoa do grupo revisa antes do merge.

```bash
git switch main && git pull
git switch -c feature/minha-feature
# ... código, scripts, testes ...
pytest
git push -u origin feature/minha-feature   # e abra o PR no GitHub
```

## Documentação da API

As funções públicas usam docstrings em português brasileiro no estilo NumPy. O gerador é o [pdoc](https://pdoc.dev/):

```bash
python scripts/build_docs.py
```

Abra `docs/api/index.html` no navegador.

No interpretador:

```python
from drilling.features.well_path import solve_case1, DEFAULT_WELL_PATH_INPUT
from drilling.core import parse_vector3, WellPathInput

help(solve_case1)
help(parse_vector3)
```
