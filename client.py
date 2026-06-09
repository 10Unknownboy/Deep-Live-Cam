#!/usr/bin/env python3
import argparse
import asyncio
import logging
import time
import sys
import threading
from fractions import Fraction
import cv2
import numpy as np
import aiohttp
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration, RTCIceServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dlc-client")

class WebcamTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        super().__init__()
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened() and sys.platform == "win32":
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")
        self._fps = fps
        self._n = 0
        
    async def recv(self):
        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.cap.read)
        if not ret:
            raise EOFError("Camera disconnected")
            
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts = self._n
        new_frame.time_base = Fraction(1, self._fps)
        self._n += 1
        
        # Explicitly yield control to prevent event loop starvation
        await asyncio.sleep(0.001)
        return new_frame

    def stop(self):
        super().stop()
        if self.cap:
            self.cap.release()

async def run_webrtc_client(url, camera, width, height, fps, use_vcam, vw, vh, turn_url, turn_user, turn_pass):
    while True:
        ice = []
        if turn_url:
            ice.append(RTCIceServer(urls=[turn_url], username=turn_user, credential=turn_pass))
        else:
            ice.append(RTCIceServer(urls=["stun:stun.l.google.com:19302"]))
            ice.append(RTCIceServer(urls=["turn:openrelay.metered.ca:80"], username="openrelayproject", credential="openrelayproject"))
            ice.append(RTCIceServer(urls=["turn:openrelay.metered.ca:443"], username="openrelayproject", credential="openrelayproject"))
            ice.append(RTCIceServer(urls=["turn:openrelay.metered.ca:443?transport=tcp"], username="openrelayproject", credential="openrelayproject"))

        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice))
        cam_track = WebcamTrack(camera, width, height, fps)
        pc.addTrack(cam_track)

        vcam_dev = None
        frames_count = 0
        t0 = time.time()
        current_fps = 0.0

        @pc.on("track")
        async def on_track(track):
            nonlocal vcam_dev, frames_count, t0, current_fps
            logger.info(f"Remote track received: {track.kind}")
            
            while True:
                try:
                    f = await track.recv()
                    img = f.to_ndarray(format="bgr24")
                    
                    frames_count += 1
                    now = time.time()
                    if now - t0 >= 1.0:
                        current_fps = frames_count / (now - t0)
                        frames_count = 0
                        t0 = now
                        
                    if use_vcam:
                        if vcam_dev is None:
                            import pyvirtualcam
                            vcam_dev = pyvirtualcam.Camera(width=vw, height=vh, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
                        vcam_dev.send(cv2.resize(img, (vw, vh)))
                        vcam_dev.sleep_until_next_frame()
                    else:
                        disp = img.copy()
                        cv2.putText(disp, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Deep-Live-Cam WebRTC", disp)
                        if cv2.waitKey(1) == ord('q'):
                            break
                            
                except Exception as e:
                    logger.error(f"Track receive error: {e}")
                    break

        @pc.on("iceconnectionstatechange")
        async def on_ice():
            logger.info(f"ICE State: {pc.iceConnectionState}")
            if pc.iceConnectionState == "failed":
                logger.warning("ICE failed. Attempting ICE Restart...")
                pc.restartIce()
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(f"{url}/offer", json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}) as resp:
                            ans = await resp.json()
                            await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
                except Exception as e:
                    logger.error(f"ICE Restart signaling failed: {e}")

        # Initial Negotiation
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        try:
            logger.info(f"Connecting to {url}/offer")
            async with aiohttp.ClientSession() as session:
                hdrs = {"ngrok-skip-browser-warning": "true"}
                async with session.post(f"{url}/offer", json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}, headers=hdrs) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Server error: {await resp.text()}")
                    ans = await resp.json()
                    await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
        except Exception as e:
            logger.error(f"Signaling failed: {e}. Retrying in 5s...")
            cam_track.stop()
            await pc.close()
            await asyncio.sleep(5)
            continue

        while pc.connectionState not in ("failed", "closed"):
            await asyncio.sleep(1)

        logger.info("Connection lost. Reconnecting in 3 seconds...")
        cam_track.stop()
        await pc.close()
        cv2.destroyAllWindows()
        if vcam_dev:
            vcam_dev.close()
        await asyncio.sleep(3)

def main():
    ap = argparse.ArgumentParser(description="Deep-Live-Cam WebRTC Client")
    ap.add_argument("--url", required=True, help="Colab ngrok URL")
    ap.add_argument("--camera", type=int, default=0, help="Camera index")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--vcam", action="store_true")
    ap.add_argument("--vcam-width", type=int, default=1280)
    ap.add_argument("--vcam-height", type=int, default=720)
    ap.add_argument("--turn-url", default="")
    ap.add_argument("--turn-user", default="")
    ap.add_argument("--turn-pass", default="")
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging")
    a = ap.parse_args()

    if a.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    try:
        asyncio.run(run_webrtc_client(a.url.rstrip("/"), a.camera, a.width, a.height, a.fps, a.vcam, a.vcam_width, a.vcam_height, a.turn_url, a.turn_user, a.turn_pass))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
