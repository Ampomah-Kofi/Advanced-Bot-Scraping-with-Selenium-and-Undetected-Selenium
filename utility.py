import pyautogui
import pydirectinput
import time
import random

def find_element(image, confidence=0.8, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            pos = pyautogui.locateCenterOnScreen(image, confidence=confidence)
            if pos:
                return pos
        except Exception:
            pass
        time.sleep(0.5)
    return None

def find_and_click(image, confidence=0.8, timeout=10):
    pos = find_element(image, confidence, timeout)
    if pos:
        pydirectinput.moveTo(int(pos.x), int(pos.y), duration=0.3)
        pydirectinput.click()
        time.sleep(1)
        return True
    return False

def find_and_click_retry(image, confidence=0.7, retries=15, interval=1.0):
    for _ in range(retries):
        try:
            pos = pyautogui.locateCenterOnScreen(image, confidence=confidence)
            if pos:
                pydirectinput.moveTo(int(pos.x), int(pos.y))
                time.sleep(0.2)
                pydirectinput.moveRel(2, 0)
                time.sleep(0.1)
                pydirectinput.click()
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False
