#!/usr/bin/env python3
"""Deep-Live-Cam WebSocket Client

Streams your local webcam to a remote Deep-Live-Cam Colab server
via WebSockets and displays the face-swapped result in real time.

Usage:
    pip install aiohttp opencv-python numpy pyvirtualcam
    python client.py --url https://xxxx.ngrok-free.app
"""

import argparse
import asyncio
import logging
import time
import aiohttp
import cv2
import numpy as np

logger = logging.getLogger("dlc-client")

async def run_client(url, camera, w, h, fps, use_vcam, vw, vh, retries, delay):
    for attempt in range(1, retries + 1):
        if attempt > 1:
            logger.info(f"Retry {attempt}/{retries} in {delay}s...")
            await asyncio.sleep(delay)
        try:
            await _stream(url, camera, w, h, fps, use_vcam, vw, vh)
            return
        except KeyboardInterrupt:
            return
        except Exception as e:
            logger.error(f"Connection error: {e}")
    logger.error(f"Max retries ({retries}) exceeded.")

async def _stream(url, camera, w, h, fps, use_vcam, vw, vh):
    ws_url = url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera}")
        
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera opened: {aw}x{ah}")
    
    vcam_dev = None
    if use_vcam:
        try:
            import pyvirtualcam
            vcam_dev = pyvirtualcam.Camera(width=vw, height=vh, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
            logger.info(f"Virtual cam: {vcam_dev.device}")
        except Exception as e:
            logger.warning(f"VCam unavailable: {e}")

    logger.info(f"Connecting to {ws_url} ...")
    
    async with aiohttp.ClientSession() as session:
        # ngrok-skip-browser-warning avoids the interstitial HTML page on free Ngrok accounts
        hdrs = {"ngrok-skip-browser-warning": "true"}
        async with session.ws_connect(ws_url, headers=hdrs, max_msg_size=10*1024*1024) as ws:
            logger.info("Connected!")
            print("\n" + "=" * 50)
            print("  Deep-Live-Cam Client Running (WebSocket)")
            print("  'q' = quit  |  'm' = mirror")
            print("=" * 50 + "\n")
            
            mirror = False
            frames_count = 0
            
            try:
                while True:
                    # Clear buffer to get the freshest frame
                    for _ in range(3): 
                        cap.grab()
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.error("Camera read failed")
                        break
                        
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    
                    t0 = time.time()
                    await ws.send_bytes(buf.tobytes())
                    
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        t1 = time.time()
                        dt = t1 - t0
                        fps_val = 1.0 / dt if dt > 0 else 0
                        frames_count += 1
                        
                        arr = np.frombuffer(msg.data, np.uint8)
                        out_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        
                        disp = cv2.flip(out_frame, 1) if mirror else out_frame.copy()
                        cv2.putText(disp, f"Ping: {dt*1000:.0f}ms | FPS: {fps_val:.1f} | #{frames_count}", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.imshow("Deep-Live-Cam", disp)
                        if vcam_dev:
                            try:
                                vcam_dev.send(cv2.resize(out_frame, (vw, vh)))
                                vcam_dev.sleep_until_next_frame()
                            except Exception:
                                pass
                                
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        logger.error("WebSocket closed by server.")
                        break
                        
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("m"):
                        mirror = not mirror
                        print(f"Mirror: {'ON' if mirror else 'OFF'}")
                        
            finally:
                cv2.destroyAllWindows()
                cap.release()
                if vcam_dev:
                    vcam_dev.close()

def main():
    ap = argparse.ArgumentParser(description="Deep-Live-Cam WebSocket Client")
    ap.add_argument("--url", required=True, help="Colab ngrok URL")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--width", type=int, default=640, help="Capture width")
    ap.add_argument("--height", type=int, default=480, help="Capture height")
    ap.add_argument("--fps", type=int, default=30, help="Capture FPS")
    ap.add_argument("--vcam", action="store_true", help="Enable virtual camera")
    ap.add_argument("--vcam-width", type=int, default=1280, help="VCam width")
    ap.add_argument("--vcam-height", type=int, default=720, help="VCam height")
    ap.add_argument("--retries", type=int, default=10, help="Max retries")
    ap.add_argument("--retry-delay", type=float, default=3.0, help="Retry delay")
    ap.add_argument("--verbose", action="store_true", help="Enable debug")
    a = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
                        
    try:
        asyncio.run(run_client(a.url, a.camera, a.width, a.height, a.fps,
                               a.vcam, a.vcam_width, a.vcam_height,
                               a.retries, a.retry_delay))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
