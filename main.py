"""
Opens top.gg bot pages, clicks Invite, waits for Discord OAuth,
clicks Continue, clicks Authorize, then closes the tab.
"""

import pandas as pd
import subprocess
import time
import os
import pyautogui
from utility import find_and_click, find_and_click_retry

# ============================================================================
# CONFIGURATION
# ============================================================================
csv_file = "bot_commands_final66.csv"

CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

DELAY_PAGE_LOAD = 7
DELAY_AFTER_INVITE = 6
DELAY_AFTER_CONTINUE = 4
DELAY_NEXT_BOT = 2

INVITE_IMAGE = "images/invite.png"
CONTINUE_IMAGE = "images/continue.png"
AUTHORIZE_IMAGE = "images/authorize.png"

# ============================================================================
# CHROME HELPERS
# ============================================================================
def find_chrome():
    for path in CHROME_PATHS:
        if os.path.exists(path):
            return path
    return None

def open_chrome_tab(url, chrome_path, first=False):
    if first:
        subprocess.Popen([chrome_path, "--window-size=1920,1080", url])
    else:
        subprocess.Popen([chrome_path, "--new-tab", url])

def close_tab():
    pyautogui.hotkey("ctrl", "w")
    time.sleep(1)

def focus_browser():
    pyautogui.click(500, 500)
    time.sleep(0.5)

# ============================================================================
# MAIN
# ============================================================================
def main():
    chrome = find_chrome()
    if not chrome:
        print("Chrome not found")
        return

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} bots")

    for idx, row in df.iterrows():
        bot_id = row["bot_id"]
        bot_name = row["name"]
        url = f"https://top.gg/bot/{bot_id}"

        print(f"\n[{idx + 1}/{len(df)}] {bot_name}")

        try:
            # Step 1. Open bot page
            open_chrome_tab(url, chrome, first=(idx == 0))
            time.sleep(DELAY_PAGE_LOAD)
            focus_browser()

            # Step 2. Click Invite
            if not find_and_click(INVITE_IMAGE, confidence=0.6, timeout=10):
                print("Invite not found")
                close_tab()
                continue

            print("Invite clicked")
            time.sleep(DELAY_AFTER_INVITE)
            focus_browser()

            # Step 3. Wait for Continue and click
            print("Waiting for Continue to activate")
            if not find_and_click_retry(
                CONTINUE_IMAGE,
                confidence=0.6,
                retries=15,
                interval=1.0
            ):
                print("Continue not activated")
                close_tab()
                continue

            print("Continue clicked")
            time.sleep(DELAY_AFTER_CONTINUE)
            focus_browser()

            # Step 4. Wait for Authorize and click
            print("Waiting for Authorize to activate")
            if not find_and_click_retry(
                AUTHORIZE_IMAGE,
                confidence=0.6,
                retries=15,
                interval=1.0
            ):
                print("Authorize not activated")
                close_tab()
                continue

            print("Authorization successful")
            time.sleep(4)

            # Close tab and move on
            close_tab()
            time.sleep(DELAY_NEXT_BOT)

        except KeyboardInterrupt:
            print("Stopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            close_tab()
            time.sleep(DELAY_NEXT_BOT)

    print("Finished")

if __name__ == "__main__":
    main()
