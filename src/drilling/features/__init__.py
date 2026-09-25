"""Correção de trajetória e otimização de trajetória Tipo 1.

O aplicativo desktop expõe dois módulos independentes:

* ``drilling.features.well_path`` — correção de trajetória 3D (Casos 1–3).
* ``drilling.features.minimization`` — otimização de trajetória Tipo 1 em 2D.

Os núcleos matemáticos ficam nesses pacotes. Tipos, parse e envelopes de
dados ficam em ``drilling.core``. A casca Qt fica em ``drilling.gui``.
"""
