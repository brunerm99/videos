# pulse_compression.py

import sys

from manim import *
from MF_Tools import VT
from scipy.interpolate import interp1d

sys.path.insert(0, "..")
from props import WeatherRadarTower, get_blocks
from props.style import BACKGROUND_COLOR, IF_COLOR, RX_COLOR, TX_COLOR

config.background_color = BACKGROUND_COLOR

SKIP_ANIMATIONS_OVERRIDE = False

FONT = "Maple Mono CN"

BLOCKS = get_blocks()


def skip_animations(b):
    return b and (not SKIP_ANIMATIONS_OVERRIDE)


def fh(scene, scale=1):
    return scene.camera.frame.height * scale


def fw(scene, scale=1):
    return scene.camera.frame.width * scale
