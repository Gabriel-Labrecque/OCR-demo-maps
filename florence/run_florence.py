from worker import run_florence
import sys

if __name__ == "__main__":
    image_path = sys.argv[1]
    run_florence(image_path)