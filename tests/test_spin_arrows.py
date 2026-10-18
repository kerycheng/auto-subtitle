import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QStyle, QStyleOptionSpinBox

from autosub.appearance import DESKTOP_STYLE, DecimalBox, NumberBox, prepare_desktop_theme


def button_rect(box, control):
    option = QStyleOptionSpinBox()
    box.initStyleOption(option)
    return box.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option, control, box)


def glyph_color(box, control):
    image = box.grab().toImage()
    point = button_rect(box, control).center()
    ratio = image.devicePixelRatio()
    return image.pixelColor(round(point.x() * ratio), round(point.y() * ratio)).name()


@pytest.mark.parametrize("kind,value,step", [(NumberBox, 5, 1), (DecimalBox, .5, .05)])
def test_visible_arrows_match_mouse_and_keyboard_steps(qtbot, qapp, kind, value, step):
    prepare_desktop_theme(qapp)
    box = kind()
    qtbot.addWidget(box)
    box.setStyleSheet(DESKTOP_STYLE)
    box.setRange(0, 10)
    box.setSingleStep(step)
    box.setValue(value)
    box.resize(180, 36)
    box.show()
    up = button_rect(box, QStyle.SubControl.SC_SpinBoxUp)
    down = button_rect(box, QStyle.SubControl.SC_SpinBoxDown)
    assert up.width() >= 20 and down.width() >= 20
    assert not up.intersects(down)
    assert glyph_color(box, QStyle.SubControl.SC_SpinBoxUp) == "#52657e"
    assert glyph_color(box, QStyle.SubControl.SC_SpinBoxDown) == "#52657e"
    qtbot.mouseClick(box, Qt.MouseButton.LeftButton, pos=up.center())
    assert box.value() == pytest.approx(value + step)
    qtbot.mouseClick(box, Qt.MouseButton.LeftButton, pos=down.center())
    assert box.value() == pytest.approx(value)
    qtbot.keyClick(box, Qt.Key.Key_Up)
    assert box.value() == pytest.approx(value + step)
    qtbot.keyClick(box, Qt.Key.Key_Down)
    assert box.value() == pytest.approx(value)
    box.setMinimum(value)
    assert glyph_color(box, QStyle.SubControl.SC_SpinBoxDown) == "#a6b0bd"
    qtbot.mouseClick(box, Qt.MouseButton.LeftButton, pos=down.center())
    assert box.value() == pytest.approx(value)
    box.setMaximum(value)
    assert glyph_color(box, QStyle.SubControl.SC_SpinBoxUp) == "#a6b0bd"
    box.setEnabled(False)
    assert glyph_color(box, QStyle.SubControl.SC_SpinBoxDown) == "#a6b0bd"
