import qrcode

# Replace with your actual GitHub repository URL
github_url = "https://github.com"

# Configure the QR code structure
qr = qrcode.QRCode(
    version=1,
    box_size=10,
    border=4,
)
qr.add_data(github_url)
qr.make(fit=True)

# Create and save the image
img = qr.make_image(fill_color="black", back_color="white")
img.save("github_repo_qr.png")
print("QR Code generated and saved successfully!")
