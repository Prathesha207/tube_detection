import os
from PIL import Image

def generate_ico():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(current_dir, "icon.png")
    ico_path = os.path.join(current_dir, "icon.ico")
    
    if not os.path.exists(png_path):
        print(f"Error: {png_path} not found")
        return
        
    img = Image.open(png_path)
    # Standard Windows icon sizes: 16, 24, 32, 48, 64, 128, 256
    icon_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(ico_path, format="ICO", sizes=icon_sizes)
    print(f"Generated multi-resolution icon: {ico_path}")

if __name__ == "__main__":
    generate_ico()
