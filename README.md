# OpenVINO Face Detection Auto Switch

Real-time face detection system using OpenVINO with dynamic device switching between CPU GPU and AUTO along with a live GUI monitoring panel.

---

## Features

- Real-time face detection using OpenVINO
- Dynamic device switching CPU GPU AUTO
- GUI Control Panel using Tkinter
- Live FPS monitoring
- CPU usage tracking
- Thread-safe inference handling
- Optimized performance with OpenVINO runtime

---

## Demo

Press buttons in control panel to switch device in real-time.

- CPU mode stable performance
- GPU mode accelerated inference if available
- AUTO mode automatic device selection

---

## Project Structure
 openvino-face-detection-auto-switch/
 |
 |---app.py
 |---requirements.txt
 |---README.md
 |---.gitignore 
  
  
---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/uvv-01/openvino-face-detection-auto-switch.git
cd openvino-face-detection-auto-switch
  

2. Create virtual environment 
  python -m venv venv  

3. Activate virtual environment
  .\venv\scripts\Activate
   
   windows (CMD)
   venv\Scripts\activate 

4. Install dependencies 
   
   pip  install -r requirements.txt

  run the project  
     python app.py 


🎮 Controls
Use GUI buttons to switch devices (CPU / GPU / AUTO)
Press ESC in camera window to exit  

  📦 Requirements
Python 3.9+
OpenVINO (2024+)
OpenCV
psutil
tkinter   
  

  📦 Requirements
Python 3.9+
OpenVINO (2024+)
OpenCV
psutil
tkinter     

   🔮 Future Improvements
Auto load balancing between CPU and GPU
Performance graphs and analytics
Support for multiple models
Web dashboard using FastAPI
Docker support   
   
 Author  
   Yuvraj singh 
   GitHub: https://github.com/uvv-01  



📜 License

This project is open source and available under the MIT License.
   


 


