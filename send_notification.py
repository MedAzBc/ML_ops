import subprocess
import sys

def send_notification(title, message):
    # Use PowerShell to send a notification
    subprocess.run(['powershell.exe', '-Command', 
                    f'New-BurntToastNotification -Text "{title}", "{message}"'])

if __name__ == "__main__":
    title = sys.argv[1]
    message = sys.argv[2]
    send_notification(title, message)

