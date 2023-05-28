from flask import Flask, request, render_template
import torch
from torch import nn
import base64
import io
from PIL import Image

app = Flask(__name__)

noise_dim = 64

def get_gen_block(in_channels, out_channels, kernel_size, stride, final_block=False):
    if final_block:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.Tanh()
        )
    else:  
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        self.block_1 = get_gen_block(noise_dim, 256, (3, 3), 2)
        self.block_2 = get_gen_block(256, 128, (4, 4), 1)
        self.block_3 = get_gen_block(128, 64, (3, 3), 2)
        self.block_4 = get_gen_block(64, 1, (4, 4), 2, final_block=True)

    def forward(self, r_noise_vec):
        x = r_noise_vec.view(-1, self.noise_dim, 1, 1)
        x1 = self.block_1(x)
        x2 = self.block_2(x1)
        x3 = self.block_3(x2)
        x4 = self.block_4(x3)
        return x4

# Load the generator model
G = Generator(noise_dim)
G.load_state_dict(torch.load("generator_model.pth", map_location=torch.device("cpu")))
G.eval()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the number of images to generate from the form data
        num_images = int(request.form["num_images"])

        # Generate images using the loaded generator
        noise_dim = 64  # Assuming the noise dimension is 64
        noise = torch.randn(num_images, noise_dim)
        generated_images = G(noise).detach().cpu()

        # Convert generated images to base64-encoded strings
        generated_images_base64 = []
        for image in generated_images:
            image_pil = Image.fromarray(image.squeeze().numpy() * 255).convert("L")
            buffer = io.BytesIO()
            image_pil.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            generated_images_base64.append(image_base64)

        return render_template("index.html", generated_images=generated_images_base64)

    return render_template("index.html")


if __name__ == "__main__":
    app.run()
