import streamlit as st
import requests
import numpy as np
from PIL import Image
import PIL.ImageOps
import pandas as pd
import altair as alt

SERVICE_URL = "http://localhost:3000/predict"

def main():
    st.set_page_config(layout="centered")
    st.title('Fashion MNIST Image Classification')
    st.write('Upload an image of a fashion item and let the model predict the class.')

    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with col2:
            st.image(image, caption='Uploaded Image', use_container_width=True)

        image = image.resize((28, 28))
        image = PIL.ImageOps.invert(image.convert('RGB'))
        gray_image = np.array(image.convert('L'))
        pixel_values = gray_image.flatten().tolist()

        try:
            response = requests.post(SERVICE_URL, json={"image": pixel_values})

            if response.status_code == 200:
                result = response.json()
                st.write(f"Predicted class ID: {result['class_id']}")
                st.write(f"Predicted class name: {result['class_name']}")
                
                # Create DataFrame directly
                df = pd.DataFrame(result['probabilities'].items(), columns=['Class', 'Probability'])
                
                # Simple Altair chart with labels
                base = alt.Chart(df).encode(
                    x=alt.X('Probability', axis=alt.Axis(format='%')),
                    y=alt.Y('Class', sort='-x'),
                    text=alt.Text('Probability', format='.1%')
                )
                
                # Combine bar chart and text labels
                chart = base.mark_bar() + base.mark_text(align='left', dx=2)
                
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write(f"Error: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.ConnectionError:
            st.write("Error: Could not connect to the service.")
            st.write("Make sure the service is running with: bentoml serve service:FashionMNISTService")


if __name__ == "__main__":
    main()