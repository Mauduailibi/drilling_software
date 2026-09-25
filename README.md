# Drilling Software

Software desktop (PySide6) com dois módulos independentes:

- **Well Path Correction** (`drilling.features.well_path`) — correção de trajetória 3D (Cases 1, 2 e 3).
- **Minimization** (`drilling.features.minimization`) — otimização de trajetória Tipo 1 (força, torque, tempo de broca, tempo total) sobre um modelo geológico 3D (`geology.HorizonModel`: horizontes inclinados, falhas e fácies laterais, ou seja, litologias diferentes na mesma profundidade).

A matemática dos solvers está congelada pelos testes golden. Qualquer mudança de fórmula, limite ou default da GUI deve falhar em `pytest`.

Tipos compartilhados (`Point3D`, `Point2D`, `WellPathInput`, `ConstraintCheck`) e o parse de campos `x, y[, z]` ficam em `drilling.core`. Os dois módulos continuam independentes e usam convenções de eixo diferentes (Z negativo × Z positivo): **não há conversão automática entre eles**.

## Coordenadas (não misturar)

| Módulo | Eixos | Profundidade |
|--------|--------|----------------|
| Well Path | XYZ, metros | **Z negativo** |
| Minimization | XYZ, metros (poço no plano vertical P0 → P3) | **Z positivo** para baixo |

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
# Rápido: Cases 1–3, ponto L1=1200 m / R=500 m, ótimos mecânicos
pytest

# Completo: também os quatro objetivos no grid L1 × R (~15 min)
pytest -m slow
```

Regenerar snapshots **somente** se a mudança numérica for intencional:

```bash
python scripts/capture_goldens.py        # rápido
python scripts/capture_goldens.py --slow # inclui os 4 ótimos
```

## Scripts de pesquisa

Scripts de pesquisa (não são testes) ficam em `scripts/` e usam o mesmo pacote da GUI. É o lugar para testar e visualizar resultados rapidamente, sem abrir a interface. Arquivos gerados vão para `outputs/` (ignorado pelo git).

```bash
python scripts/optimization_demo.py      # quatro objetivos + gráficos, geologia 3D de exemplo
python scripts/selected_trajectory.py    # inspeciona um par (L1, R) escolhido
python scripts/scenario_3d.py            # compara geologia plana, inclinada e com fácies lateral

# Sensibilidade de malha (ΔL1, ΔR, passo do elemento)
python scripts/sensitivity/post_std.py --run all
python scripts/sensitivity/post_corrigido_min_l1.py --run all   # idem, restrito a L1 >= min_l1
python scripts/sensitivity/post2.py                              # compara ΔL1 = 10 m e 1 m (lê outputs/post_outputs)
python scripts/sensitivity/post2_torque.py
```

Os modelos geológicos de exemplo (`flat`, `dipping`, `facies`) ficam em `scripts/example_meshes.py`.

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
