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
# Tudo, inclusive os quatro objetivos no grid L1 × R (~20 s)
pytest

# Sem as varreduras completas do grid
pytest -m "not slow"
```

`tests/test_geology.py` valida o modelo geológico 3D: heterogeneidade lateral, regressão das camadas planas contra a implementação antiga, soft-string contra a solução fechada e o filtro de manobras por litologia.

Regenerar snapshots **somente** se a mudança numérica for intencional:

```bash
python scripts/capture_goldens.py        # rápido
python scripts/capture_goldens.py --slow # inclui os 4 ótimos (alguns minutos)
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
