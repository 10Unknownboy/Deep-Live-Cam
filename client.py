#!/usr/bin/env python3
"""Deep-Live-Cam WebRTC Client

Streams your local webcam to a remote Deep-Live-Cam Colab server
via WebRTC and displays the face-swapped result in real time.

Usage:
    pip install aiortc aiohttp opencv-python av pyvirtualcam
    python client.py --url https://xxxx.ngrok-free.app
    python client.py --url https://xxxx.ngrok-free.app --camera 1 --vcam
"""

import argparse
import asyncio
import logging
import sys
import time
import threading
from fractions import Fraction

import aiohttp
import cv2
import numpy as np
from av import VideoFrame
from aiortc import (
    RTCPeerConnection, RTCSessionDescription,
    MediaStreamTrack, RTCConfiguration, RTCIceServer,
)

logger = logging.getLogger("dlc-client")


class WebcamTrack(MediaStreamTrack):
    """Captures frames from local webcam and sends via WebRTC."""
    kind = "video"

    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        af = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {aw}x{ah} @ {af}fps")
        self._time_base = Fraction(1, fps)
        self._pts = 0

    async def recv(self):
        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.cap.read)
        if not ret:
            raise RuntimeError("Failed to read from camera")
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts = self._pts
        vf.time_base = self._time_base
        self._pts += 1
        return vf

    def stop(self):
        super().stop()
        if self.cap:
            self.cap.release()


class FrameReceiver:
    """Collects frames from remote processed video track."""
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.count = 0
        self._ts = []
        self.fps = 0.0

    def update(self, bgr):
        with self.lock:
            self.frame = bgr
            self.count += 1
            now = time.time()
            self._ts.append(now)
            if len(self._ts) > 30:
                self._ts = self._ts[-30:]
            if len(self._ts) >= 2:
                dt = self._ts[-1] - self._ts[0]
                if dt > 0:
                    self.fps = (len(self._ts) - 1) / dt

    def get(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


async def run_client(url, camera, w, h, fps, vcam, vw, vh, retries, delay):
    """Main loop with auto-reconnect."""
    for attempt in range(1, retries + 1):
        if attempt > 1:
            logger.info(f"Retry {attempt}/{retries} in {delay}s...")
            await asyncio.sleep(delay)
        try:
            await _stream(url, camera, w, h, fps, vcam, vw, vh)
            return
        except KeyboardInterrupt:
            return
        except Exception as e:
            logger.error(f"Connection error: {e}")
    logger.error(f"Max retries ({retries}) exceeded.")


async def _stream(url, camera, w, h, fps, use_vcam, vw, vh):
    """Connect, stream, display."""
    rx = FrameReceiver()
    cfg = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
    pc = RTCPeerConnection(configuration=cfg)
    cam = WebcamTrack(camera, w, h, fps)
    pc.addTrack(cam)

    @pc.on("track")
    def on_track(track):
        logger.info(f"Remote track: {track.kind}")
        if track.kind == "video":
            asyncio.ensure_future(_consume(track, rx))

    @pc.on("connectionstatechange")
    async def on_state():
        logger.info(f"State: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            raise ConnectionError("WebRTC lost")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    offer_url = url.rstrip("/") + "/offer"
    logger.info(f"Sending offer to {offer_url}")

    async with aiohttp.ClientSession() as sess:
        hdrs = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
        payload = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        async with sess.post(offer_url, json=payload, headers=hdrs, ssl=False) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Server {resp.status}: {await resp.text()}")
            ans = await resp.json()

    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
    logger.info("Connected!")

    vcam_dev = None
    if use_vcam:
        try:
            import pyvirtualcam
            vcam_dev = pyvirtualcam.Camera(width=vw, height=vh, fps=fps,
                                           fmt=pyvirtualcam.PixelFormat.BGR)
            logger.info(f"Virtual cam: {vcam_dev.device}")
        except Exception as e:
            logger.warning(f"VCam unavailable: {e}")

    print("\n" + "=" * 50)
    print("  Deep-Live-Cam Client Running")
    print("  'q' = quit  |  'm' = mirror")
    print("=" * 50 + "\n")

    mirror = False
    try:
        while True:
            f = rx.get()
            if f is not None:
                disp = cv2.flip(f, 1) if mirror else f.copy()
                cv2.putText(disp, f"FPS: {rx.fps:.1f} | #{rx.count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Deep-Live-Cam", disp)
                if vcam_dev:
                    try:
                        vcam_dev.send(cv2.resize(f, (vw, vh)))
                        vcam_dev.sleep_until_next_frame()
                    except Exception:
                        pass
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                mirror = not mirror
                print(f"Mirror: {'ON' if mirror else 'OFF'}")
            await asyncio.sleep(0.001)
    finally:
        cv2.destroyAllWindows()
        cam.stop()
        if vcam_dev:
            vcam_dev.close()
        await pc.close()


async def _consume(track, rx):
    """Consume frames from the remote video track."""
    while True:
        try:
            frame = await track.recv()
            rx.update(frame.to_ndarray(format="bgr24"))
        except Exception:
            break


def main():
    ap = argparse.ArgumentParser(description="Deep-Live-Cam WebRTC Client")
    ap.add_argument("--url", required=True, help="Colab ngrok URL")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    ap.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    ap.add_argument("--fps", type=int, default=30, help="Capture FPS (default: 30)")
    ap.add_argument("--vcam", action="store_true", help="Enable virtual camera output")
    ap.add_argument("--vcam-width", type=int, default=1280, help="Virtual camera width")
    ap.add_argument("--vcam-height", type=int, default=720, help="Virtual camera height")
    ap.add_argument("--retries", type=int, default=10, help="Max reconnection attempts")
    ap.add_argument("--retry-delay", type=float, default=3.0, help="Delay between retries (s)")
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging")
    a = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    print("\n" + "=" * 50)
    print("  🎭 Deep-Live-Cam WebRTC Client")
    print("=" * 50)
    print(f"  Server:  {a.url}")
    print(f"  Camera:  {a.camera} ({a.width}x{a.height} @ {a.fps}fps)")
    print(f"  VCam:    {'ON' if a.vcam else 'OFF'}")
    print("=" * 50 + "\n")

    try:
        asyncio.run(run_client(a.url, a.camera, a.width, a.height, a.fps,
                               a.vcam, a.vcam_width, a.vcam_height,
                               a.retries, a.retry_delay))
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
