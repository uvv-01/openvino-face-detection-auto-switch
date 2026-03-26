import cv2
import time
import psutil
import threading
import tkinter as tk
from openvino.runtime import Core

# ================== OPENVINO SETUP ==================
ie = Core()
model_path = "intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
model = ie.read_model(model=model_path)

available_devices = ie.available_devices
print("Available devices:", available_devices)

current_device = "CPU"
compiled_model = ie.compile_model(model=model, device_name=current_device)

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# ================== GLOBAL VARIABLES ==================
running = True
fps = 0
cpu_usage = 0
last_switch_time = 0

lock = threading.Lock()

# ================== DEVICE SWITCH ==================
def switch_device(device):
    global compiled_model, input_layer, output_layer, current_device

    if device not in available_devices and device != "AUTO":
        print(f"❌ {device} not available")
        return

    print(f"🔄 Switching to {device}...")

    try:
        with lock:
            new_model = ie.compile_model(model=model, device_name=device)

            compiled_model = new_model
            input_layer = compiled_model.input(0)
            output_layer = compiled_model.output(0)
            current_device = device

        print(f"✅ Switched to {device}")

    except Exception as e:
        print("❌ Switch failed:", e)


# ================== AUTO SMART SWITCH ==================
def auto_switch():
    global last_switch_time

    while running:
        time.sleep(3)

        if time.time() - last_switch_time < 5:
            continue

        # CPU overloaded OR FPS low → GPU
        if current_device == "CPU":
            if cpu_usage > 80 or fps < 15:
                print("⚡ AUTO: CPU overloaded → GPU")
                switch_device("GPU")
                last_switch_time = time.time()

        # System stable → back to CPU
        elif current_device == "GPU":
            if cpu_usage < 40 and fps > 20:
                print("⚡ AUTO: Stable → CPU")
                switch_device("CPU")
                last_switch_time = time.time()


# ================== VIDEO LOOP ==================
def video_loop():
    global fps, cpu_usage, running

    print("🎥 Video thread started")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Camera not opening")
        return

    prev_time = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Preprocess
        resized = cv2.resize(frame, (672, 384))
        input_data = resized.transpose((2, 0, 1))
        input_data = input_data.reshape(1, 3, 384, 672)

        # SAFE INFERENCE
        with lock:
            result = compiled_model([input_data])[output_layer]

        # Draw detections
        for detection in result[0][0]:
            if detection[2] > 0.5:
                xmin = int(detection[3] * w)
                ymin = int(detection[4] * h)
                xmax = int(detection[5] * w)
                ymax = int(detection[6] * h)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=None)

        # Display info
        cv2.putText(frame, f"FPS: {int(fps)}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.putText(frame, f"Device: {current_device}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.putText(frame, f"CPU: {cpu_usage}%", (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("OpenVINO Face Detection", frame)

        # ESC to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Video stopped")


# ================== GUI ==================
def start_gui():
    global running

    root = tk.Tk()
    root.title("OpenVINO Control Panel")
    root.geometry("320x350")

    tk.Label(root, text="Device Control", font=("Arial", 14)).pack(pady=10)

    tk.Button(root, text="CPU", width=15,
              command=lambda: switch_device("CPU")).pack(pady=5)

    tk.Button(root, text="GPU", width=15,
              command=lambda: switch_device("GPU")).pack(pady=5)

    tk.Button(root, text="AUTO", width=15,
              command=lambda: switch_device("AUTO")).pack(pady=5)

    # Monitor labels
    fps_label = tk.Label(root, text="FPS: 0")
    fps_label.pack(pady=10)

    cpu_label = tk.Label(root, text="CPU Usage: 0%")
    cpu_label.pack(pady=10)

    device_label = tk.Label(root, text="Device: CPU")
    device_label.pack(pady=10)

    def update_labels():
        fps_label.config(text=f"FPS: {int(fps)}")
        cpu_label.config(text=f"CPU Usage: {cpu_usage}%")
        device_label.config(text=f"Device: {current_device}")
        root.after(500, update_labels)

    update_labels()

    def on_close():
        global running
        running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


# ================== RUN ==================
video_thread = threading.Thread(target=video_loop, daemon=True)
auto_thread = threading.Thread(target=auto_switch, daemon=True)

video_thread.start()
auto_thread.start()

start_gui()

running = False
video_thread.join()