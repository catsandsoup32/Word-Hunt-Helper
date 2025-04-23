from picamera2 import Picamera2, Preview
import time


picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
picam2.configure(camera_config)
picam2.start_preview(Preview.QTGL)
picam2.start()
counter = 14
while True:
    input("waiting for user input")
    picam2.capture_file(f"{counter}.jpg")
    counter += 1