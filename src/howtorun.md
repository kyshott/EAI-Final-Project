<div align="center">

# How to run

For starters, create a virtual environment after cloning this repo or downloading the **src** folder. Then, install all package dependencies listed in requirements.txt (make sure to use **pip install -r** for this)

To run the scripts provided, ensure that a stable microphone is plugged in and working to your device. Additionally, ensure that, on edge devices, there is some free memory as running these scripts can potentially introduce
some issues. Next, simply run the scripts using the "Python" command in a terminal, and observe the results. The **inf-metrics** file uses dummy data, so a microphone is not required. It will simply perform inference, and report results.
The same goes for the **TTFI** script, which will do the same. Finally, the **live-inference.py** file does require a fully functioning microphone to work. Run the script, and observe the reconstruction MSEs in the terminal. 
Ensure that a fan is running near the microphone, and adjust the threshold as needed. 
