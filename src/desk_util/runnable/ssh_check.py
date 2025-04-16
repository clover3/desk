import socket
import time
import argparse
import subprocess
import platform
from datetime import datetime


def check_ssh(host, port=22, timeout=5):
    """Check if SSH port is open on the target host."""
    try:
        socket_obj = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket_obj.settimeout(timeout)
        result = socket_obj.connect_ex((host, port))
        socket_obj.close()
        return result == 0
    except Exception as e:
        print(f"Error checking connection: {e}")
        return False


def notify(title, message):
    """Send desktop notification based on the operating system."""
    system = platform.system()

    try:
        if system == 'Darwin':  # macOS
            subprocess.run(['osascript', '-e', f'display notification "{message}" with title "{title}"'])
            # Also play a sound
            subprocess.run(['afplay', '/System/Library/Sounds/Ping.aiff'])
        elif system == 'Linux':
            subprocess.run(['notify-send', title, message])
        elif system == 'Windows':
            import winsound
            winsound.Beep(1000, 500)  # Frequency: 1000Hz, Duration: 500ms
            # Method 2: Using Windows native MessageBox
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, message, title, 0x40)

        print(f"\n{title}: {message}")
    except Exception as e:
        print(f"Could not send notification: {e}")


def monitor_ssh(host, port=22, interval=10, timeout=5):
    """Monitor a host for SSH availability and notify when it becomes available."""
    print(f"Starting SSH monitor for {host}:{port}")
    print(f"Will check every {interval} seconds...")
    print("Press Ctrl+C to stop")

    try:
        attempt = 1
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if check_ssh(host, port, timeout):
                notify("SSH Available", f"Server {host}:{port} is now responding to SSH!")
                print(f"\n[{timestamp}] Attempt {attempt}: SSH is AVAILABLE on {host}:{port}")
                break
            else:
                print(
                    f"[{timestamp}] Attempt {attempt}: SSH not available on {host}:{port}. Checking again in {interval} seconds...",
                    end="\r")

            attempt += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor a server for SSH availability")
    default_host = "login.hpc.virginia.edu"
    # default_host = "isb.or.kr"
    # Make host optional with nargs='?' and provide a default value
    parser.add_argument("host", nargs='?', default=default_host,
                        help=f"Hostname or IP address to monitor (default: {default_host})")
    parser.add_argument("-p", "--port", type=int, default=22,
                        help="SSH port (default: 22)")
    parser.add_argument("-i", "--interval", type=int, default=10,
                        help="Check interval in seconds (default: 10)")
    parser.add_argument("-t", "--timeout", type=int, default=5,
                        help="Connection timeout in seconds (default: 5)")

    args = parser.parse_args()
    monitor_ssh(args.host, args.port, args.interval, args.timeout)