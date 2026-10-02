"""Camada de carregamento que cobre uma aba enquanto um cálculo roda em outra thread."""

from __future__ import annotations

import time

from PySide6.QtCore import QEvent, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class LoadingOverlay(QWidget):
    """Cobre o widget pai com um véu translúcido, um spinner e o tempo decorrido.

    Enquanto visível, a camada recebe os cliques do mouse, então nada embaixo
    dela pode ser usado. Chame ``start(mensagem)`` e ``stop()``.
    """

    SPINNER_RADIUS = 26
    SPINNER_WIDTH = 5

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.message = ""
        self.angle = 0
        self.started_at = 0.0
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        parent.installEventFilter(self)
        self.hide()

    def start(self, message: str) -> None:
        self.message = message
        self.started_at = time.monotonic()
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()
        self.timer.start()

    def stop(self) -> None:
        self.timer.stop()
        self.hide()

    def elapsed_text(self) -> str:
        seconds = int(time.monotonic() - self.started_at)
        return f"{seconds // 60}:{seconds % 60:02d}"

    def _tick(self) -> None:
        self.angle = (self.angle + 6) % 360
        self.update()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.parentWidget() and event.type() == QEvent.Resize:
            self.setGeometry(self.parentWidget().rect())
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event) -> None:
        event.accept()

    def wheelEvent(self, event) -> None:
        event.accept()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(255, 255, 255, 200))

        center = self.rect().center()
        r = self.SPINNER_RADIUS
        box = QRectF(center.x() - r, center.y() - r - 24, 2 * r, 2 * r)
        pen = QPen(QColor("#e5e7eb"), self.SPINNER_WIDTH)
        painter.setPen(pen)
        painter.drawEllipse(box)
        pen.setColor(QColor("#2563eb"))
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        # Ângulos do Qt em 1/16 de grau, sentido anti-horário.
        painter.drawArc(box, -self.angle * 16, 100 * 16)

        painter.setPen(QColor("#111827"))
        font = QFont(self.font())
        font.setPointSize(font.pointSize() + 3)
        painter.setFont(font)
        text_top = int(box.bottom()) + 18
        painter.drawText(QRectF(0, text_top, self.width(), 28), Qt.AlignHCenter, self.message)
        font.setPointSize(font.pointSize() - 2)
        painter.setFont(font)
        painter.setPen(QColor("#6b7280"))
        painter.drawText(QRectF(0, text_top + 30, self.width(), 24), Qt.AlignHCenter, f"Elapsed {self.elapsed_text()}")
