from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from manimlib import ImageMobject
from PIL import Image


def _parse_frame_rate(value):
    try:
        numerator, denominator = value.split("/")
        return float(numerator) / float(denominator)
    except (AttributeError, ValueError, ZeroDivisionError):
        return 0.0


def _probe_video(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,r_frame_rate,duration:"
        "stream_tags=alpha_mode:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in {path}")

    stream = streams[0]
    duration = float(stream.get("duration", 0) or 0)
    if duration <= 0:
        duration = float(data.get("format", {}).get("duration", 0) or 0)

    frame_rate = _parse_frame_rate(stream.get("r_frame_rate"))
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width <= 0 or height <= 0 or frame_rate <= 0 or duration <= 0:
        raise ValueError(f"Could not read complete video metadata from {path}")

    alpha_mode = str(stream.get("tags", {}).get("alpha_mode", "")).lower()
    decoder = ""
    if stream.get("codec_name") == "vp9" and alpha_mode in ("1", "true", "yes"):
        decoder = "libvpx-vp9"

    return width, height, frame_rate, duration, decoder


def _update_video(video, dt):
    video._update_frame(dt)


class VideoMobject(ImageMobject):
    def __init__(
        self,
        filename,
        speed=1.0,
        autoplay=True,
        start=0.0,
        end=None,
        **kwargs,
    ):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("VideoMobject requires ffmpeg and ffprobe")

        self.video_path = Path(filename).expanduser().resolve()
        if not self.video_path.is_file():
            raise FileNotFoundError(self.video_path)

        (
            self.frame_width,
            self.frame_height,
            self.frame_rate,
            self.source_duration,
            self._decoder_name,
        ) = _probe_video(self.video_path)

        self.start = float(start)
        self.end = self.source_duration if end is None else float(end)
        if not np.isfinite(self.start) or self.start < 0:
            raise ValueError("start must be nonnegative and finite")
        if self.start >= self.source_duration:
            raise ValueError("start must be less than the video duration")
        if not np.isfinite(self.end) or self.end <= self.start:
            raise ValueError("end must be finite and greater than start")
        if self.end > self.source_duration:
            raise ValueError("end must not exceed the video duration")

        self.duration = self.end - self.start
        self._has_explicit_end = end is not None
        self.speed = 1.0
        self.set_speed(speed)
        self.paused = not autoplay
        self._time = 0.0
        self._frame_index = -1
        self._ended = False
        self._decoder = None
        self._owns_decoder = True
        self._pending_frame = None

        self._start_decoder()
        first_frame = self._read_frame()
        if first_frame is None:
            self._close_decoder()
            raise ValueError(f"Could not decode the first frame of {self.video_path}")

        self._temp_directory = tempfile.TemporaryDirectory(prefix="melp_video_")
        first_frame_path = Path(self._temp_directory.name) / "first_frame.png"
        Image.fromarray(first_frame, "RGBA").save(first_frame_path)

        super().__init__(str(first_frame_path), **kwargs)
        self._pending_frame = first_frame
        self.add_updater(_update_video, call=False)

    def _start_decoder(self):
        self._close_decoder()
        command = ["ffmpeg", "-v", "error"]
        if self._decoder_name:
            command.extend(["-c:v", self._decoder_name])
        if self.start > 0:
            command.extend(["-ss", str(self.start)])
        command.extend(
            [
                "-i",
                str(self.video_path),
            ]
        )
        if self._has_explicit_end:
            command.extend(["-t", str(self.duration)])
        command.extend(
            [
                "-an",
                "-sn",
                "-dn",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "-",
            ]
        )
        self._decoder = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._frame_index = -1

    def _close_decoder(self):
        decoder = getattr(self, "_decoder", None)
        if decoder is None or not getattr(self, "_owns_decoder", False):
            return

        if decoder.stdout is not None:
            decoder.stdout.close()
        if decoder.poll() is None:
            decoder.terminate()
            try:
                decoder.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                decoder.kill()
                decoder.wait()
        self._decoder = None

    def _read_frame(self):
        if self._decoder is None or self._decoder.stdout is None:
            return None

        frame_size = self.frame_width * self.frame_height * 4
        raw_frame = self._decoder.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            return None

        self._frame_index += 1
        return np.frombuffer(raw_frame, dtype=np.uint8).reshape(
            self.frame_height,
            self.frame_width,
            4,
        )

    def _upload_frame(self, frame):
        self.image = Image.fromarray(frame, "RGBA")
        if self.shader_wrapper is None or not self.shader_wrapper.textures:
            self._pending_frame = frame
            return

        self.shader_wrapper.textures[0].write(frame.tobytes())
        self._pending_frame = None

    def _update_frame(self, dt):
        if self._pending_frame is not None:
            self._upload_frame(self._pending_frame)
        if self.paused or self._ended or dt <= 0:
            return

        self._time = min(self._time + dt * self.speed, self.duration)
        target_frame = int(self._time * self.frame_rate)
        latest_frame = None
        while self._frame_index < target_frame:
            next_frame = self._read_frame()
            if next_frame is None:
                if latest_frame is not None:
                    self._upload_frame(latest_frame)
                self._time = self.duration
                self._ended = True
                self.paused = True
                self._close_decoder()
                return
            latest_frame = next_frame

        if latest_frame is not None:
            self._upload_frame(latest_frame)

    def _restart(self):
        self._time = 0.0
        self._ended = False
        self._start_decoder()
        first_frame = self._read_frame()
        if first_frame is None:
            self._ended = True
            self.paused = True
            self._close_decoder()
            return
        self._upload_frame(first_frame)

    def play(self, speed=None):
        if speed is not None:
            self.set_speed(speed)
        if self._ended:
            self._restart()
        if self._ended:
            return self
        self.paused = False
        return self

    def pause(self):
        self.paused = True
        return self

    def set_speed(self, speed):
        speed = float(speed)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be positive and finite")
        self.speed = speed
        return self

    def get_duration(self):
        return self.duration

    def get_time_remaining(self):
        return max(self.duration - self._time, 0.0)

    def copy(self, deep=False):
        result = super().copy(deep=deep)
        if isinstance(result, VideoMobject):
            result.clear_updaters()
            result._decoder = None
            result._owns_decoder = False
            result.paused = True
        return result

    def __del__(self):
        self._close_decoder()
