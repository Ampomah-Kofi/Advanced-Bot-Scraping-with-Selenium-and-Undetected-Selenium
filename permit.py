import pandas as pd

# Discord permission flags
PERMISSION_FLAGS = {
    1 << 0: "CREATE_INSTANT_INVITE",
    1 << 1: "KICK_MEMBERS",
    1 << 2: "BAN_MEMBERS",
    1 << 3: "ADMINISTRATOR",
    1 << 4: "MANAGE_CHANNELS",
    1 << 5: "MANAGE_GUILD",
    1 << 6: "ADD_REACTIONS",
    1 << 7: "VIEW_AUDIT_LOG",
    1 << 10: "VIEW_CHANNEL",
    1 << 11: "SEND_MESSAGES",
    1 << 12: "SEND_TTS_MESSAGES",
    1 << 13: "MANAGE_MESSAGES",
    1 << 14: "EMBED_LINKS",
    1 << 15: "ATTACH_FILES",
    1 << 16: "READ_MESSAGE_HISTORY",
    1 << 17: "MENTION_EVERYONE",
    1 << 18: "USE_EXTERNAL_EMOJIS",
    1 << 20: "CONNECT",
    1 << 21: "SPEAK",
    1 << 22: "MUTE_MEMBERS",
    1 << 23: "DEAFEN_MEMBERS",
    1 << 24: "MOVE_MEMBERS",
    1 << 25: "USE_VAD",
    1 << 27: "CHANGE_NICKNAME",
    1 << 28: "MANAGE_NICKNAMES",
    1 << 29: "MANAGE_ROLES",
    1 << 30: "MANAGE_WEBHOOKS",
    1 << 31: "MANAGE_EMOJIS"
}

def decode_permissions(permission_value):
    """Decode numeric Discord permissions into readable names."""
    try:
        value = int(permission_value)
        perms = [name for bit, name in PERMISSION_FLAGS.items() if value & bit]
        return ", ".join(perms) if perms else "NONE"
    except Exception:
        return "INVALID"

# Load dataset
df = pd.read_csv(r"c:\Users\kampomah\Desktop\Bot data Cleaning\bot_commands_final26.csv")

# Decode permissions
df["decoded_permissions"] = df["permissions"].apply(decode_permissions)

# Save to new CSV
df.to_csv("bot_permissions_decoded_full.csv", index=False)

print("✅ Permissions decoded and saved to bot_permissions_decoded_full.csv")
