import streamlit as st
from diffusers import DiffusionPipeline
from PIL import Image
from io import BytesIO
import os

# Load the model
model_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id)
pipeline.enable_model_cpu_offload()

# Streamlit App
st.title("Text to Image Generation")
st.write("Generate images from text using the DisneyPixarCartoon768 model.")

# Text input from user
user_input = st.text_input("Enter a description for the image:")

if st.button("Generate Image"):
    with st.spinner("Generating..."):
        # Generate image
        result = pipeline(user_input)
        image = result.images[0]

        # Ensure the image is in the correct format
        image = image.convert("RGB")

        # Save the image to a BytesIO object for Streamlit
        img_bytes = BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        # Display the image in Streamlit
        st.image(img_bytes, caption=user_input, use_column_width=True)

        # Pop up the image on the local computer
        temp_image_path = "temp_generated_image.png"
        image.save(temp_image_path)  # Save image temporarily
        image.show()  # This will open the image in the default viewer

        # Clean up the temporary file if needed
        os.remove(temp_image_path)

    # Log user input
    print(f"Generated image for: {user_input}")
