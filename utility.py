import pyautogui
import pydirectinput
import time
import random

def find_element(image_path, confidence=0.8, timeout=10):
    """Find an image on screen using OpenCV and return its position."""
    start = time.time()
    while time.time() - start < timeout:
        # ADDED grayscale=True here. This helps it ignore color changes!
        pos = pyautogui.locateCenterOnScreen(image_path, confidence=confidence, grayscale=True)
        if pos:
            print(f"[FOUND] {image_path} at {pos}")
            return pos
        time.sleep(0.5)
    print(f"[MISS] {image_path}")
    return None

def find_and_click(image_path, confidence=0.8, timeout=10):
    """Find and click an element."""
    pos = find_element(image_path, confidence, timeout)
    if pos:
        x, y = pos
        pydirectinput.moveTo(x, y, duration=0.3)
        pydirectinput.click()
        time.sleep(1)
        return True
    return False

def type_write(text):
    """Type text into input field."""
    for c in text:
        pydirectinput.typewrite(c)
        time.sleep(random.uniform(0.02, 0.08))