import streamlit as st
import subprocess
import tempfile
import os

st.title('YOLOv6 Object Detection with Streamlit')

weights_path = 'YOLOv6-main\\yolov6s6.pt'  # Path to YOLOv6 weights file
source_image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png', 'webp'])

if source_image is not None:
    # Create a temporary directory to store the uploaded image
    temp_dir = tempfile.TemporaryDirectory()
    temp_image_path = os.path.join(temp_dir.name, 'temp_image.jpg')

    # Read the content of the uploaded image and save it to the temporary location
    with open(temp_image_path, 'wb') as f:
        f.write(source_image.read())

    # Run the YOLOv6 object detection command 
    cmd = f'python .YOLOv6-main\\tools\\infer.py --weights {weights_path} --source {temp_image_path}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Display the YOLOv6 Inference Result
    st.text('YOLOv6 Inference Result:')
    # st.text(result.stdout)
    # st.text('Captured Standard Error:')
    # st.text(result.stderr)

    # Display the annotated image
    from PIL import Image
    output_image_path = 'YOLOv6-main/runs/inference/exp/temp_image.jpg'
    annotated_image = Image.open(output_image_path)
    st.image(annotated_image, caption='Annotated Image', use_column_width=True, width=300)


    # Clean up the temporary directory
    temp_dir.cleanup()
