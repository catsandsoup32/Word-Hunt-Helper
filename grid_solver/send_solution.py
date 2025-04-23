# Main file to solve the Word Hunt grid,
# then send solution to Arduino.

from picamera2 import Picamera2, Preview
from PIL import Image
import argparse

from trie import TrieNode
from solver import build_trie, solve
from character_recognition.custom_parse import get_grid

parser = argparse.ArgumentParser()
parser.add_argument("--pause_print", type=bool, required=False)
parser.add_argument("--show_process", type=bool, required=False)
args = parser.parse_args()

root = TrieNode()
build_trie(root)

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
picam2.configure(camera_config)
# picam2.start_preview(Preview.QTGL)
picam2.start()
input("waiting for user input")
image_array = picam2.capture_array()

pause_print = args.pause_print if args.pause_print is not None else False
show_process = args.show_process if args.show_process is not None else False
solutions = solve(root, pause_print, Image.open("character_recognition/test2.jpg"), show_process)

