import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import redis
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

# Read Redis connection details from environment variables
redis_host = os.getenv("REDIS_HOST")
redis_port = os.getenv("REDIS_PORT")
redis_password = os.getenv("REDIS_PASSWORD")

redis_client = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Attendance Tracker")

        self.tree = ttk.Treeview(self.root, columns=('Student Name', 'Timestamp'))
        self.tree.heading('#0', text='Student Name')
        self.tree.heading('#1', text='Timestamp')
        self.tree.pack(padx=10, pady=10)

        self.refresh_button = tk.Button(self.root, text='Refresh', command=self.refresh_attendance)
        self.refresh_button.pack(pady=10)

        # Setup the automatic refresh
        self.setup_automatic_refresh()

    def refresh_attendance(self):
        print("refreshing")
        # Get the current time
        current_time = datetime.now()

        # Set the time threshold for the last 5 minutes
        threshold_time = current_time - timedelta(minutes=1)

        # Get log entries from Redis within the last 5 minutes
        latest_log_entries = self.get_latest_log_entries(threshold_time)

        # Clear existing items in the treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Display the latest log entries in the treeview
        for log_entry in latest_log_entries:
            self.tree.insert('', 'end', values=(log_entry['student_name'], log_entry['timestamp']))

    def get_latest_log_entries(self, threshold_time):
        latest_log_entries = {}
        current_time = datetime.now()

        if redis_client.exists('attendance:logs'):
            log_entries = redis_client.lrange('attendance:logs', 0, -1)

            for log_entry_str in log_entries:
                try:
                    parts = log_entry_str.split('%')
                    student_name = parts[0]
                    timestamp_str = parts[-1]
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")

                    if (current_time - timestamp).total_seconds() <= timedelta(minutes=1).total_seconds():
                        # Update the dictionary only if this record is more recent
                        if student_name not in latest_log_entries or timestamp > latest_log_entries[student_name]['timestamp']:
                            latest_log_entries[student_name] = {'student_name': student_name, 'timestamp': timestamp}

                except Exception as e:
                    print(f"Error processing log entry: {log_entry_str}")
                    print(f"Exception: {e}")

        return list(latest_log_entries.values())
    
    
    def setup_automatic_refresh(self):
        # Refresh every 60 seconds (60000 milliseconds)
        self.refresh_attendance()  # Perform initial refresh
        self.root.after(60000, self.setup_automatic_refresh)

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
