import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from manim import *
from PIL import Image, ImageOps


def pixel_format_has_alpha(pixel_format):
    if not pixel_format:
        return False

    normalized = pixel_format.lower()
    alpha_prefixes = (
        "rgba",
        "bgra",
        "argb",
        "abgr",
        "yuva",
        "gbrap",
        "ya",
        "ayuv",
    )
    return normalized.startswith(alpha_prefixes)


def parse_frame_rate(frame_rate_text):
    try:
        numerator, denominator = frame_rate_text.split("/")
        denominator_value = float(denominator)
        if denominator_value == 0:
            return 30.0
        return float(numerator) / denominator_value
    except (AttributeError, ValueError, ZeroDivisionError):
        return 30.0


def probe_video_stream(filename):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt,r_frame_rate",
        "-of",
        "json",
        str(filename),
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        streams = json.loads(result.stdout).get("streams", [])
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None

    if not streams:
        return None

    stream = streams[0]
    return {
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "pix_fmt": stream.get("pix_fmt", ""),
        "frame_rate": parse_frame_rate(stream.get("r_frame_rate", "30/1")),
    }


@dataclass
class VideoStatus:
    time: float = 0
    videoObject: Any = None
    frame_rate: float = 30.0
    frame_width: int = 0
    frame_height: int = 0
    frame_index: int = -1
    ffmpeg_process: Any = None
    use_ffmpeg_pipe: bool = False
    last_frame: Any = None
    paused: bool = False

    def __deepcopy__(self, memo):
        return self


class VideoMobject(ImageMobject):
    """
    Following a discussion on Discord about animated GIF images.
    Modified for videos
    Parameters
    ----------
    filename
        the filename of the video file
    imageops
        (optional) possibility to include a PIL.ImageOps operation, e.g.
        PIL.ImageOps.mirror
    speed
        (optional) speed-up/slow-down the playback
    loop
        (optional) replay the video from the start in an endless loop
    https://discord.com/channels/581738731934056449/1126245755607339250/1126245755607339250
    2023-07-06 Uwe Zimmermann & Abulafia
    2024-03-09 Uwe Zimmermann
    """

    @staticmethod
    def frame_to_image(frame):
        if frame.ndim == 2:
            return Image.fromarray(frame)

        channel_count = frame.shape[2]

        if channel_count == 4:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA))

        if channel_count == 3:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if channel_count == 1:
            return Image.fromarray(frame[:, :, 0])

        return Image.fromarray(frame)

    @staticmethod
    def open_ffmpeg_pipe(filename):
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(filename),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgra",
            "-",
        ]
        try:
            return subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return None

    def close_ffmpeg_pipe(self):
        process = self.status.ffmpeg_process
        if process is None:
            return

        stdout = process.stdout
        if stdout is not None:
            stdout.close()

        process.kill()
        process.wait()
        self.status.ffmpeg_process = None

    def restart_ffmpeg_pipe(self):
        self.close_ffmpeg_pipe()
        self.status.ffmpeg_process = self.open_ffmpeg_pipe(self.filename)
        self.status.frame_index = -1
        self.status.last_frame = None
        return self.status.ffmpeg_process is not None

    def read_ffmpeg_frame(self):
        process = self.status.ffmpeg_process
        if process is None or process.stdout is None:
            return False, None

        frame_size = self.status.frame_width * self.status.frame_height * 4
        raw_frame = process.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            return False, None

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
            (self.status.frame_height, self.status.frame_width, 4)
        )
        frame = frame.copy()
        self.status.frame_index += 1
        self.status.last_frame = frame
        return True, frame

    def read_frame(self):
        if self.status.use_ffmpeg_pipe:
            target_frame = max(int(self.status.time * self.status.frame_rate / 1000), 0)

            if target_frame < self.status.frame_index:
                if not self.restart_ffmpeg_pipe():
                    return False, None

            while self.status.frame_index < target_frame:
                ret, frame = self.read_ffmpeg_frame()
                if not ret:
                    return False, None

            if self.status.last_frame is None:
                return self.read_ffmpeg_frame()

            return True, self.status.last_frame

        if self.status.videoObject is None:
            return False, None

        self.status.videoObject.set(cv2.CAP_PROP_POS_MSEC, self.status.time)
        ret, frame = self.status.videoObject.read()
        return ret, frame

    def pause(self):
        self.status.paused = True
        return self

    def resume(self):
        self.status.paused = False
        return self

    def play(self):
        return self.resume()

    def toggle_pause(self):
        self.status.paused = not self.status.paused
        return self

    def __init__(self, filename=None, imageops=None, speed=1.0, loop=False, **kwargs):
        if filename is None:
            raise ValueError("VideoMobject requires a filename")

        self.filename = Path(filename)
        self.imageops = imageops
        self.speed = speed
        self.loop = loop
        self._id = id(self)
        self.status = VideoStatus()

        metadata = probe_video_stream(self.filename)
        if metadata is not None:
            self.status.frame_rate = metadata["frame_rate"]
            self.status.frame_width = metadata["width"]
            self.status.frame_height = metadata["height"]
            self.status.use_ffmpeg_pipe = pixel_format_has_alpha(metadata["pix_fmt"])

        ret = False
        frame = None
        if self.status.use_ffmpeg_pipe and self.restart_ffmpeg_pipe():
            ret, frame = self.read_ffmpeg_frame()

        if not ret:
            self.status.use_ffmpeg_pipe = False
            self.close_ffmpeg_pipe()
            self.status.videoObject = cv2.VideoCapture(str(self.filename))
            self.status.videoObject.set(cv2.CAP_PROP_POS_FRAMES, 1)
            ret, frame = self.status.videoObject.read()

        if ret:
            img = self.frame_to_image(frame)

            if imageops != None:
                img = imageops(img)
        else:
            img = Image.fromarray(
                np.array(
                    [[63, 0, 0, 0], [0, 127, 0, 0], [0, 0, 191, 0], [0, 0, 0, 255]],
                    dtype=np.uint8,
                )
            )
        super().__init__(np.asarray(img), **kwargs)
        if ret:
            self.add_updater(self.videoUpdater)

    def videoUpdater(self, mobj, dt):
        if dt == 0:
            return
        status = self.status
        if status.paused:
            return

        status.time += 1000 * dt * mobj.speed

        ret, frame = self.read_frame()
        if (ret == False) and self.loop:
            status.time = 0
            if status.use_ffmpeg_pipe:
                if self.restart_ffmpeg_pipe():
                    ret, frame = self.read_ffmpeg_frame()
            elif self.status.videoObject is not None:
                self.status.videoObject.set(cv2.CAP_PROP_POS_MSEC, status.time)
                ret, frame = self.status.videoObject.read()

        if ret:
            img = self.frame_to_image(frame)

            if mobj.imageops != None:
                img = mobj.imageops(img)
            mobj.pixel_array = change_to_rgba_array(
                np.asarray(img), mobj.pixel_array_dtype
            )

    def __del__(self):
        video_object = getattr(self.status, "videoObject", None)
        if video_object is not None:
            video_object.release()

        self.close_ffmpeg_pipe()
