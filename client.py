#!/usr/bin/env python3
import argparse
import asyncio
import logging
import time
import sys
import threading
import cv2
import numpy as np
import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dlc-client")

async def ws_stream(url, camera, w, h, fps, use_vcam, vw, vh):
    ws_url = url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened() and sys.platform == "win32":
        cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    
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
            
    async with aiohttp.ClientSession() as session:
        hdrs = {"ngrok-skip-browser-warning": "true"}
        logger.info(f"Connecting to {ws_url} ...")
        async with session.ws_connect(ws_url, headers=hdrs, max_msg_size=10*1024*1024) as ws:
            logger.info("Connected!")
            print("\n" + "=" * 50)
            print("  Deep-Live-Cam Client Running (Decoupled WebSocket)")
            print("  'q' = quit  |  'm' = mirror")
            print("=" * 50 + "\n")
            
            mirror = False
            
            async def sender():
                # Throttle sender to 15 FPS max to prevent TCP buffer bloat over Ngrok
                send_fps = min(fps, 15)
                while not ws.closed:
                    ret, frame = await asyncio.get_event_loop().run_in_executor(None, cap.read)
                    if not ret: break
                    
                    # Downscale and compress heavily to guarantee low latency over free tunnels
                    h, w = frame.shape[:2]
                    if max(h, w) > 480:
                        scale = 480 / max(h, w)
                        send_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                    else:
                        send_frame = frame
                        
                    _, buf = cv2.imencode(".jpg", send_frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
                    try:
                        await ws.send_bytes(buf.tobytes())
                    except Exception:
                        break
                    await asyncio.sleep(1.0 / send_fps)
                    
            async def receiver():
                nonlocal mirror
                frames_count = 0
                t0 = time.time()
                fps_val = 0.0
                while not ws.closed:
                    try:
                        msg = await ws.receive()
                    except Exception:
                        break
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        frames_count += 1
                        now = time.time()
                        if now - t0 >= 1.0:
                            fps_val = frames_count / (now - t0)
                            frames_count = 0
                            t0 = now
                            
                        arr = np.frombuffer(msg.data, np.uint8)
                        out_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        disp = cv2.flip(out_frame, 1) if mirror else out_frame.copy()
                        cv2.putText(disp, f"FPS: {fps_val:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Deep-Live-Cam", disp)
                        
                        if vcam_dev:
                            try:
                                vcam_dev.send(cv2.resize(out_frame, (vw, vh)))
                                vcam_dev.sleep_until_next_frame()
                            except: pass
                            
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            await ws.close()
                            break
                        elif key == ord('m'):
                            mirror = not mirror
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

            await asyncio.gather(sender(), receiver())
            
    cap.release()
    cv2.destroyAllWindows()
    if vcam_dev: vcam_dev.close()

async def run_client(url, camera, w, h, fps, use_vcam, vw, vh, retries, delay):
    for attempt in range(1, retries + 1):
        if attempt > 1:
            logger.info(f"Retry {attempt}/{retries} in {delay}s...")
            await asyncio.sleep(delay)
        try:
            await ws_stream(url, camera, w, h, fps, use_vcam, vw, vh)
            return
        except KeyboardInterrupt:
            return
        except Exception as e:
            logger.error(f"Connection error: {e}")
    logger.error(f"Max retries ({retries}) exceeded.")

def main():
    ap = argparse.ArgumentParser(description="Deep-Live-Cam Decoupled WebSocket Client")
    ap.add_argument("--url", required=True, help="Colab ngrok URL")
    ap.add_argument("--camera", type=int, default=0, help="Camera index")
    ap.add_argument("--width", type=int, default=640, help="Capture width")
    ap.add_argument("--height", type=int, default=480, help="Capture height")
    ap.add_argument("--fps", type=int, default=15, help="Capture FPS")
    ap.add_argument("--vcam", action="store_true", help="Enable virtual camera")
    ap.add_argument("--vcam-width", type=int, default=1280, help="VCam width")
    ap.add_argument("--vcam-height", type=int, default=720, help="VCam height")
    ap.add_argument("--retries", type=int, default=10, help="Max retries")
    ap.add_argument("--retry-delay", type=float, default=3.0, help="Retry delay")
    ap.add_argument("--verbose", action="store_true", help="Enable debug")
    a = ap.parse_args()

    if a.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    try:
        asyncio.run(run_client(a.url, a.camera, a.width, a.height, a.fps, a.vcam, a.vcam_width, a.vcam_height, a.retries, a.retry_delay))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
