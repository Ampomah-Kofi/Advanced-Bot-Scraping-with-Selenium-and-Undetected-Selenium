"""
Discord Bot Permission Privacy Analyzer - Complete Edition
Analyzes permissions and generates both report and detailed statistics
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict

# ============================================================================
# DISCORD PERMISSION DEFINITIONS
# ============================================================================

DISCORD_PERMISSIONS = {
    0x0000000001: ('CREATE_INSTANT_INVITE', 'low', 'Can create invite links'),
    0x0000000002: ('KICK_MEMBERS', 'high', 'Can kick members - moderation power'),
    0x0000000004: ('BAN_MEMBERS', 'high', 'Can ban members - moderation power'),
    0x0000000008: ('ADMINISTRATOR', 'critical', 'Full server control - DANGEROUS'),
    0x0000000010: ('MANAGE_CHANNELS', 'medium', 'Can modify channels'),
    0x0000000020: ('MANAGE_GUILD', 'high', 'Can modify server settings'),
    0x0000000040: ('ADD_REACTIONS', 'low', 'Can add reactions'),
    0x0000000080: ('VIEW_AUDIT_LOG', 'medium', 'Can view server audit logs'),
    0x0000000100: ('PRIORITY_SPEAKER', 'low', 'Priority speaker in voice'),
    0x0000000200: ('STREAM', 'low', 'Can stream in voice'),
    0x0000000400: ('VIEW_CHANNEL', 'low', 'Can view channels'),
    0x0000000800: ('SEND_MESSAGES', 'low', 'Can send messages'),
    0x0000001000: ('SEND_TTS_MESSAGES', 'low', 'Can send text-to-speech'),
    0x0000002000: ('MANAGE_MESSAGES', 'medium', 'Can delete others\' messages'),
    0x0000004000: ('EMBED_LINKS', 'low', 'Can embed links'),
    0x0000008000: ('ATTACH_FILES', 'low', 'Can attach files'),
    0x0000010000: ('READ_MESSAGE_HISTORY', 'medium', 'Can read message history'),
    0x0000020000: ('MENTION_EVERYONE', 'medium', 'Can mention @everyone'),
    0x0000040000: ('USE_EXTERNAL_EMOJIS', 'low', 'Can use external emojis'),
    0x0000080000: ('VIEW_GUILD_INSIGHTS', 'medium', 'Can view server analytics'),
    0x0000100000: ('CONNECT', 'low', 'Can connect to voice'),
    0x0000200000: ('SPEAK', 'low', 'Can speak in voice'),
    0x0000400000: ('MUTE_MEMBERS', 'high', 'Can mute members in voice'),
    0x0000800000: ('DEAFEN_MEMBERS', 'medium', 'Can deafen members in voice'),
    0x0001000000: ('MOVE_MEMBERS', 'medium', 'Can move members between voice'),
    0x0002000000: ('USE_VAD', 'low', 'Can use voice activity'),
    0x0004000000: ('CHANGE_NICKNAME', 'low', 'Can change own nickname'),
    0x0008000000: ('MANAGE_NICKNAMES', 'medium', 'Can change others\' nicknames'),
    0x0010000000: ('MANAGE_ROLES', 'high', 'Can manage roles - POWERFUL'),
    0x0020000000: ('MANAGE_WEBHOOKS', 'high', 'Can manage webhooks'),
    0x0040000000: ('MANAGE_EMOJIS', 'medium', 'Can manage emojis'),
    0x0080000000: ('USE_SLASH_COMMANDS', 'low', 'Can use slash commands'),
}

PRIVACY_SENSITIVE_PERMISSIONS = {
    'READ_MESSAGE_HISTORY': 'Can read all past messages (privacy concern)',
    'VIEW_AUDIT_LOG': 'Can see admin actions and user activity',
    'VIEW_GUILD_INSIGHTS': 'Can access server analytics and user metrics',
    'MANAGE_MESSAGES': 'Can access and delete user messages',
    'ADMINISTRATOR': 'Full access to everything (maximum privacy risk)'
}

# ============================================================================
# CORE ANALYSIS FUNCTIONS
# ============================================================================

def decode_permissions(permission_int):
    """Decode Discord permission integer into list of permissions"""
    try:
        perms_value = int(permission_int)
    except (ValueError, TypeError):
        return []
    
    granted_permissions = []
    
    for bit_flag, (name, risk_level, description) in DISCORD_PERMISSIONS.items():
        if perms_value & bit_flag:
            granted_permissions.append({
                'name': name,
                'risk_level': risk_level,
                'description': description,
                'is_privacy_sensitive': name in PRIVACY_SENSITIVE_PERMISSIONS
            })
    
    return granted_permissions

def analyze_bot_privacy(csv_file):
    """Analyze all bots in CSV for privacy concerns"""
    df = pd.read_csv(csv_file).fillna("")
    
    results = []
    
    for _, row in df.iterrows():
        bot_id = str(row.get('bot_id', ''))
        bot_name = str(row.get('name', '')).strip()
        permissions_raw = row.get('permissions', '0')
        github_url = str(row.get('github_url', ''))
        
        if not bot_name:
            continue
        
        # Decode permissions
        permissions = decode_permissions(permissions_raw)
        
        # Count privacy concerns
        privacy_concerns = [p for p in permissions if p['is_privacy_sensitive']]
        
        # Check if open source
        is_open_source = github_url and github_url != 'nan' and 'github.com' in github_url.lower()
        
        # Calculate risk score (0-100, lower = worse privacy)
        risk_score = 100
        
        # Deduct for dangerous permissions
        risk_counts = defaultdict(int)
        for p in permissions:
            risk_counts[p['risk_level']] += 1
        
        risk_score -= risk_counts['critical'] * 40
        risk_score -= risk_counts['high'] * 20
        risk_score -= risk_counts['medium'] * 10
        risk_score -= len(privacy_concerns) * 10
        
        # Add for transparency
        if is_open_source:
            risk_score += 20
        
        risk_score = max(0, min(100, risk_score))
        
        results.append({
            'bot_name': bot_name,
            'bot_id': bot_id,
            'total_permissions': len(permissions),
            'privacy_concerns': privacy_concerns,
            'risk_score': risk_score,
            'is_open_source': is_open_source,
            'has_admin': any(p['name'] == 'ADMINISTRATOR' for p in permissions),
            'has_message_history': any(p['name'] == 'READ_MESSAGE_HISTORY' for p in permissions)
        })
    
    return results

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_summary_report(results):
    """Generate privacy research report"""
    total = len(results)
    
    admin_bots = sum(1 for r in results if r['has_admin'])
    message_history = sum(1 for r in results if r['has_message_history'])
    open_source = sum(1 for r in results if r['is_open_source'])
    with_privacy_concerns = sum(1 for r in results if len(r['privacy_concerns']) > 0)
    
    avg_risk = sum(r['risk_score'] for r in results) / total if total > 0 else 0
    
    report = f"""
{'='*70}
DISCORD BOT PERMISSION & PRIVACY ANALYSIS
{'='*70}

DATASET OVERVIEW
{'='*70}
Total Bots Analyzed: {total}
Average Privacy Score: {avg_risk:.1f}/100 (higher = better privacy)

PERMISSION ANALYSIS
{'='*70}
Bots With Administrator Permission: {admin_bots} ({admin_bots/total*100:.1f}%)
Bots With Message History Access: {message_history} ({message_history/total*100:.1f}%)
Bots With Privacy-Sensitive Permissions: {with_privacy_concerns} ({with_privacy_concerns/total*100:.1f}%)

TRANSPARENCY ANALYSIS
{'='*70}
Open Source Bots: {open_source} ({open_source/total*100:.1f}%)
Closed Source Bots: {total-open_source} ({(total-open_source)/total*100:.1f}%)

TOP 10 HIGHEST PRIVACY RISK BOTS (Lowest Scores)
{'='*70}
"""
    
    sorted_bots = sorted(results, key=lambda x: x['risk_score'])[:10]
    
    for i, bot in enumerate(sorted_bots, 1):
        report += f"\n{i}. {bot['bot_name']} (Privacy Score: {bot['risk_score']}/100)\n"
        report += f"   Total Permissions: {bot['total_permissions']}\n"
        report += f"   Privacy Concerns: {len(bot['privacy_concerns'])}\n"
        report += f"   Administrator: {'Yes' if bot['has_admin'] else 'No'}\n"
        report += f"   Message History: {'Yes' if bot['has_message_history'] else 'No'}\n"
        report += f"   Open Source: {'Yes' if bot['is_open_source'] else 'No'}\n"
        
        if bot['privacy_concerns']:
            report += f"   🚨 Privacy Issues:\n"
            for concern in bot['privacy_concerns'][:3]:
                report += f"      - {concern['name']}\n"
    
    report += f"""
KEY FINDINGS FOR RESEARCH PAPER
{'='*70}

1. EXCESSIVE PERMISSIONS
   - {admin_bots/total*100:.1f}% request Administrator (full control)
   - Users cannot make informed decisions about bot permissions
   - No principle of least privilege

2. PRIVACY RISKS
   - {message_history/total*100:.1f}% can read message history (surveillance)
   - {with_privacy_concerns/total*100:.1f}% have privacy-sensitive permissions
   - No user notification when bots access their data

3. TRANSPARENCY DEFICIT  
   - Only {open_source/total*100:.1f}% are open source
   - Users cannot verify what bots do with permissions
   - Closed-source bots create "black box" privacy risk

RECOMMENDATIONS
{'='*70}
1. Discord should show permission implications in plain language
2. Require bots to disclose data collection practices
3. Implement granular permission controls
4. Mandatory transparency reports for bots with sensitive permissions
5. User dashboard showing which bots have accessed their data

{'='*70}
"""
    
    return report

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def generate_statistical_analysis(results):
    """Generate detailed statistical analysis for research paper"""
    
    scores = [bot['risk_score'] for bot in results]
    perms = [bot['total_permissions'] for bot in results]
    concerns = [len(bot['privacy_concerns']) for bot in results]
    
    admin_bots = sum(1 for bot in results if bot['has_admin'])
    msg_history = sum(1 for bot in results if bot['has_message_history'])
    open_source = sum(1 for bot in results if bot['is_open_source'])
    
    stats_report = f"""
{'='*70}
DETAILED STATISTICAL ANALYSIS FOR RESEARCH PAPER
{'='*70}

📊 PRIVACY SCORE STATISTICS (N={len(scores)})
{'-'*70}
Mean:                {np.mean(scores):.2f}/100
Median:              {np.median(scores):.2f}/100
Standard Deviation:  {np.std(scores, ddof=1):.2f}
Minimum:             {np.min(scores):.2f}/100
Maximum:             {np.max(scores):.2f}/100
Q1 (25th percentile):{np.percentile(scores, 25):.2f}/100
Q3 (75th percentile):{np.percentile(scores, 75):.2f}/100
IQR:                 {np.percentile(scores, 75) - np.percentile(scores, 25):.2f}

📈 SCORE DISTRIBUTION
{'-'*70}
"""
    
    bins = [
        (0, 0, "Critical Risk (0)"),
        (1, 25, "High Risk (1-25)"),
        (26, 50, "Medium Risk (26-50)"),
        (51, 75, "Low Risk (51-75)"),
        (76, 100, "Minimal Risk (76-100)")
    ]
    
    for min_val, max_val, label in bins:
        if min_val == 0 and max_val == 0:
            count = sum(1 for s in scores if s == 0)
        else:
            count = sum(1 for s in scores if min_val <= s <= max_val)
        pct = count / len(scores) * 100
        stats_report += f"{label:30} {count:3} bots ({pct:5.1f}%)\n"
    
    stats_report += f"""
🔐 PERMISSION STATISTICS
{'-'*70}
Mean permissions per bot:    {np.mean(perms):.2f}
Median permissions per bot:  {np.median(perms):.2f}
Min permissions:             {np.min(perms)}
Max permissions:             {np.max(perms)}

⚠️  PRIVACY CONCERN STATISTICS
{'-'*70}
Mean privacy concerns/bot:   {np.mean(concerns):.2f}
Median privacy concerns/bot: {np.median(concerns):.2f}
Max privacy concerns:        {np.max(concerns)}

🎯 KEY METRICS FOR PAPER
{'-'*70}
Bots with ADMINISTRATOR:     {admin_bots} ({admin_bots/len(results)*100:.1f}%)
Bots with message history:   {msg_history} ({msg_history/len(results)*100:.1f}%)
Open source bots:            {open_source} ({open_source/len(results)*100:.1f}%)
Bots scoring 0/100:          {sum(1 for s in scores if s == 0)} ({sum(1 for s in scores if s == 0)/len(scores)*100:.1f}%)

📊 CORRELATIONS
{'-'*70}
"""
    
    admin_scores = [bot['risk_score'] for bot in results if bot['has_admin']]
    no_admin_scores = [bot['risk_score'] for bot in results if not bot['has_admin']]
    stats_report += f"Mean score (with ADMIN):     {np.mean(admin_scores) if admin_scores else 0:.2f}/100\n"
    stats_report += f"Mean score (without ADMIN):  {np.mean(no_admin_scores) if no_admin_scores else 0:.2f}/100\n"
    
    open_scores = [bot['risk_score'] for bot in results if bot['is_open_source']]
    closed_scores = [bot['risk_score'] for bot in results if not bot['is_open_source']]
    stats_report += f"Mean score (open source):    {np.mean(open_scores) if open_scores else 0:.2f}/100\n"
    stats_report += f"Mean score (closed source):  {np.mean(closed_scores) if closed_scores else 0:.2f}/100\n"
    
    stats_report += f"""
{'='*70}
✓ Statistical analysis complete
{'='*70}
"""
    
    return stats_report

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    import sys
    
    csv_file = "bot_commands_final66.csv"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    print("="*70)
    print("DISCORD BOT PERMISSION PRIVACY ANALYZER - COMPLETE")
    print("="*70)
    print(f"\nAnalyzing: {csv_file}\n")
    
    # Analyze
    results = analyze_bot_privacy(csv_file)
    
    if not results:
        print("✗ No bots found in CSV")
        return
    
    print(f"✓ Analyzed {len(results)} bots\n")
    
    # Generate both reports
    summary_report = generate_summary_report(results)
    stats_report = generate_statistical_analysis(results)
    
    # Combine reports
    full_report = summary_report + "\n\n" + stats_report
    
    # Save combined report
    with open("bot_permission_privacy_report_complete.txt", 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    # Print to console
    print(summary_report)
    print(stats_report)
    
    print(f"\n✓ Complete report saved: bot_permission_privacy_report_complete.txt")
    
    # Save JSON data
    with open("bot_permission_privacy_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Data saved: bot_permission_privacy_analysis.json")
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE - All outputs generated")
    print("="*70)

if __name__ == "__main__":
    main()