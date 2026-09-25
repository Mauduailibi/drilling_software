# Snapshots golden da Fase 0

Estes arquivos JSON congelam a saída numérica dos solvers atuais.

- `well_path_defaults.json` — defaults da GUI para os Casos 1–2, erro de
  alinhamento esperado do Caso 3 e um vetor viável do Caso 3.
- `minimization_point_l1_1200_r_500.json` — saídas do núcleo para uma única
  configuração Tipo 1 usada no script manual histórico.
- `minimization_mechanical_optima.json` — força mínima no topo e torque mínimo
  na malha padrão L1 × R.
- `minimization_optima.json` — quatro objetivos mais impressões digitais das
  curvas (gerado com `python scripts/capture_goldens.py --slow`).

Regenere com `scripts/capture_goldens.py`. Não edite os números à mão.
