try:
    import os
    import sys
    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle
    import cv2
    from io import BytesIO,StringIO
    from PIL import Image, ImageOps
    from keras.models import load_model
except Exception as e:
    print("Some Modules are missing...{}".format(e))
model_final = load_model('model.h5')
def main():
    st.header("Digit Recognition")
    file= st.file_uploader("Upload Image - ",type=['jpeg',"png","jpg"])
    if file is not None:
        #show_file.image(file)
        # st.write(type(file))
        file_details={
            "filename":file.name,
            "filetype":file.type,
            "filesize":file.size
        }
        #st.write(file_details)
        img=Image.open(file)
        st.image(img,width=250)
        # gr_img=ImageOps.grayscale(img).resize((28,28))
        # test_img=np.asarray(gr_img)/255.0
        test_img=np.asarray(img)/255.0
        
        
        st.write("The Recognised digit is:-")
        st.subheader(model_final.predict(test_img.reshape(1,28,28)).argmax())
    # file.close()

main()