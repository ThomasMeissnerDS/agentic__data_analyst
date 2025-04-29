from PIL import Image, ImageDraw, ImageFont
import os

def create_logo():
    # Create a new image with a white background
    image = Image.new('RGB', (200, 200), 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw a simple AI icon
    # Draw a circle
    draw.ellipse((50, 50, 150, 150), outline='blue', width=3)
    
    # Draw lines to represent AI
    draw.line((75, 75, 125, 125), fill='blue', width=3)
    draw.line((75, 125, 125, 75), fill='blue', width=3)
    
    # Save the image
    logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
    image.save(logo_path)
    print(f"Logo created at {logo_path}")

if __name__ == "__main__":
    create_logo() 