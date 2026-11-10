"""강아지 사진이 "엄마, 밥 주세요!" 라고 말하는 영상을 만든다.

사용법:
    python3 make_video.py <사진.jpg> <음성.wav> <출력.mp4> [자막]

음성의 크기(음량)에 맞춰 입(아래턱)을 벌렸다 닫았다 하고,
말풍선 자막을 한 글자씩 띄운다.
"""
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates, gaussian_filter1d

FPS = 30
W, H = 1080, 1920
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

# 사진을 1085x2012 로 줄였을 때의 입 위치 (이 사진 기준으로 맞춘 값)
MOUTH_X = 600       # 입 가운데 x
LIP_Y = 1180        # 윗입술 선 (이 위는 움직이지 않음)
JAW_Y = 1420        # 아래턱 끝
FADE_Y = 1650       # 이 아래로는 다시 움직임 없음
SIGMA_X = 150       # 입 좌우 폭
JAW_PX = 64         # 입을 최대로 벌릴 때 아래턱이 내려가는 거리(px)


def load_envelope(path, n_frames):
    with wave.open(path) as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(1)
    hop = sr / FPS
    env = np.array([
        np.sqrt(np.mean(data[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-9)
        for i in range(n_frames)
    ])
    env = env / (np.percentile(env, 98) + 1e-9)
    env = np.clip(env, 0, 1)
    env = gaussian_filter1d(env, 0.8)
    return env


def main():
    photo, audio, out = sys.argv[1:4]
    text = sys.argv[4] if len(sys.argv) > 4 else "엄마, 밥 주세요!"

    with wave.open(audio) as w:
        duration = w.getnframes() / w.getframerate()
    n = int(duration * FPS)
    env = load_envelope(audio, n)
    speaking = np.where(env > 0.08)[0]
    t0, t1 = (speaking[0], speaking[-1]) if len(speaking) else (0, n - 1)

    src = Image.open(photo).convert("RGB").resize((1085, 2012), Image.LANCZOS)
    src = np.asarray(src).astype(np.float32)
    sh, sw = src.shape[:2]

    # 입 주변만 움직이는 변형 가중치 (아래턱이 가장 많이 내려감)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    wy = np.zeros_like(yy)
    m = (yy >= LIP_Y) & (yy < JAW_Y)
    wy[m] = ((yy[m] - LIP_Y) / (JAW_Y - LIP_Y))
    m = (yy >= JAW_Y) & (yy < FADE_Y)
    wy[m] = 0.5 + 0.5 * np.cos(np.pi * (yy[m] - JAW_Y) / (FADE_Y - JAW_Y))
    wy = wy ** 1.2
    weight = wy * np.exp(-((xx - MOUTH_X) ** 2) / (2 * SIGMA_X ** 2))

    font = ImageFont.truetype(FONT, 92)
    chars = list(text)

    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", audio,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out,
    ], stdin=subprocess.PIPE)

    cx, cy = sw / 2, 1300.0
    for i in range(n):
        t = i / FPS
        e = env[i]
        # 소리가 작을 때는 입을 살짝 다물고(ㅁ, ㅂ), 클 때는 크게 벌림
        talking = t0 - 3 <= i <= t1 + 3
        open_amt = (-0.45 + 1.45 * e) if talking else 0.0
        dy = open_amt * JAW_PX * weight

        # 머리 살짝 끄덕이기 + 천천히 확대
        bob = 0.012 * e * np.sin(2 * np.pi * 3 * t) if talking else 0.0
        ang = np.deg2rad(1.2 * np.sin(2 * np.pi * 0.35 * t) + 2.0 * bob * 50)
        zoom = 1.06 + 0.04 * min(t / duration, 1.0)
        breathe = 4 * np.sin(2 * np.pi * 0.8 * t)

        # 출력 좌표 -> 원본 좌표
        oy, ox = np.mgrid[0:H, 0:W].astype(np.float32)
        ox = ox + 2.5   # 1085 -> 1080 가운데 맞춤
        px, py = ox - cx, oy - cy
        ca, sa = np.cos(ang), np.sin(ang)
        sx = (ca * px + sa * py) / zoom + cx
        sy = (-sa * px + ca * py) / zoom + cy - breathe
        # 입 변형 (원본 좌표 기준 변위를 그대로 적용)
        d = map_coordinates(dy, [sy, sx], order=1, mode="nearest")
        sy = sy - d

        frame = np.empty((H, W, 3), np.float32)
        for c in range(3):
            frame[..., c] = map_coordinates(src[..., c], [sy, sx], order=1, mode="nearest")
        img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))

        # 말풍선 + 한 글자씩 나타나는 자막
        if i >= t0 - 2:
            prog = (i - t0) / max(t1 - t0, 1)
            k = len(chars) if prog >= 1 else max(1, int(np.ceil(prog * len(chars))))
            shown = "".join(chars[:k])
            pop = min(1.0, (i - t0 + 2) / 6)
            draw_bubble(img, font, text, shown, pop)

        ff.stdin.write(img.tobytes())

    ff.stdin.close()
    ff.wait()


def draw_bubble(img, font, full, shown, pop):
    d = ImageDraw.Draw(img, "RGBA")
    bbox = d.textbbox((0, 0), full, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 60, 45
    bw, bh = (tw + 2 * pad_x) * pop, (th + 2 * pad_y) * pop
    bx, by = W / 2 - bw / 2, 190 + (1 - pop) * 40
    shadow = (bx + 8, by + 10, bx + bw + 8, by + bh + 10)
    d.rounded_rectangle(shadow, radius=int(50 * pop) + 1, fill=(0, 0, 0, 70))
    d.rounded_rectangle((bx, by, bx + bw, by + bh), radius=int(50 * pop) + 1,
                        fill=(255, 255, 255, 240), outline=(255, 140, 160, 255), width=6)
    # 말풍선 꼬리 (강아지 입 쪽으로)
    tx = W / 2 + 40
    d.polygon([(tx - 40, by + bh - 3), (tx + 30, by + bh - 3), (tx + 10, by + bh + 70 * pop)],
              fill=(255, 255, 255, 240), outline=None)
    d.line([(tx - 40, by + bh), (tx + 10, by + bh + 70 * pop), (tx + 30, by + bh)],
           fill=(255, 140, 160, 255), width=6)
    if pop >= 1:
        d.text((W / 2 - tw / 2 - bbox[0], by + pad_y - bbox[1]), shown,
               font=font, fill=(70, 50, 60, 255))


if __name__ == "__main__":
    main()
