import torch
import PIL
import json
import requests
import wget
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import torchvision.models as models
import torchvision.transforms as transforms
from IPython.display import Image, display
from trulens.nn.attribution import InputAttribution
from trulens.nn.attribution import IntegratedGradients
from trulens.visualizations import ChannelMaskVisualizer
from trulens.visualizations import MaskVisualizer
from trulens.visualizations import HeatmapVisualizer
from trulens.nn.models import get_model_wrapper
from trulens.nn.attribution import InternalInfluence
from trulens.nn.distributions import PointDoi
from trulens.nn.quantities import ClassQoI, InternalChannelQoI, MaxClassQoI
from trulens.nn.slices import Cut, InputCut, OutputCut, Slice


@st.cache
def load_keras_vis(file):
    model_builder = keras.applications.vgg16.VGG16
    img_size = (224, 224)
    preprocess_input = keras.applications.vgg16.preprocess_input
    decode_predictions = keras.applications.vgg16.decode_predictions

    last_conv_layer_name = "block5_conv3"
    
    # Make model
    model = model_builder(weights="imagenet")
    data = {}
    for layer in model.layers:
        data[str(layer.name)] = str(layer.output)
        
    data_keys = data.keys()
    data_values = data.values()
    zip_cols = {"Layer Name:": data_keys, "Layer Description": data_values}
    df1 = pd.DataFrame.from_dict(zip_cols)
    
    # Remove last layer's softmax
    model.layers[-1].activation = None
    
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(file, target_size=img_size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    img_array = np.expand_dims(array, axis=0)
    # Print what the top predicted class is
    preds = model.predict(img_array)
    predicted_string = decode_predictions(preds, top=10)[0]
    
    predicted_string_names = []
    predicted_string_scores = []
    for item in predicted_string:
        predicted_string_names.append(item[1])
        predicted_string_scores.append("{:.2f}".format(item[2]))
    
    predicted_string_data = {"Names": predicted_string_names, "Scores": predicted_string_scores}
        
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    pred_index=None
    
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    heatmap = heatmap.numpy()
    
    img = keras.preprocessing.image.load_img(file)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    
    
        
    return superimposed_img, df1, predicted_string_data

@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_model():
    # Load pre-trained version of VGG16
    pytorch_model = models.vgg16(pretrained=True)
    model = get_model_wrapper(pytorch_model, input_shape=(3,224,224), device='cpu')
    layers_dict= model._layers
    
    return pytorch_model, model, layers_dict

@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def process_image(url, pytorch_model):
    with PIL.Image.open(url) as img:
        x = np.array(img.resize((224,224), PIL.Image.ANTIALIAS))[np.newaxis]
        
        # Pre-process uploaded image
        normalize = transforms.Compose([
            transforms.ToTensor(), # convert to [0, 1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        x_pp = np.array(normalize(x[0])).transpose(1, 2, 0)[np.newaxis]

        # Transpose to [*, C, H, W] for PyTorch convention.
        x = x.transpose(0, 3, 1, 2)
        x_pp = x_pp.transpose(0, 3, 1, 2)


    # Pretty-print the model's top 5 predictions on this record.
    with torch.no_grad():
        output = pytorch_model(torch.from_numpy(x_pp).to('cpu')).cpu().numpy()

    with open('imagenet_class_index.json') as file:
        class_idx = json.load(file)
        
    idx2label = [class_idx[str(x)][1] for x in range(len(class_idx))]
    top_labels = [
        (idx, idx2label[idx], output[0][idx]) 
        for idx in np.argsort(output[0])[::-1][:10]]

    names = []
    scores = []    
    for label in top_labels:
        names.append(label[1])
        scores.append("{:.2f}".format(label[2]))
    
    predictions = {"Names": names, "Scores": scores}
    
    return img, predictions, x_pp, x


def main():
    st.set_page_config(
        page_title="Streamlit Image Classifier",
        layout="centered")
    st.markdown(
        "<font size='8'>Evaluating Image Recognition Models with Trulens",
        unsafe_allow_html=True,
    )
    
    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
    """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    
    # Produce a wrapped model from the pytorch model.
    pytorch_model, model, layers_dict = load_model()
    file = st.file_uploader("Upload your images", type=["png", "jpeg", "jpg"])
    if file:
        features_keys = layers_dict.keys()
        features_values = layers_dict.values()
        zip_cols = {"Layer Name:": features_keys, "Layer Description": features_values}
        display = pd.DataFrame.from_dict(zip_cols)
        
        heatmap, df1, predicted_string_data = load_keras_vis(file)
        
        st.markdown(
                "<font size='4'> **Model Name: PyTorch VGG16** </font>",
                unsafe_allow_html=True,
            )
        st.write(display)
        
        st.markdown(
                "<font size='4'> **Model Name: Keras VGG16** </font>",
                unsafe_allow_html=True,
            )
        st.write(df1)
        url = file
        img, label_string, x_pp, x = process_image(url, pytorch_model)
        
        st.markdown(
                "<font size='4'> **Original Image:** </font>",
                unsafe_allow_html=True,
            )
        st.image(img)
        
        x2, x3 = st.columns((2,2))
            
        with x2:
            st.markdown(
                "<font size='4'> **Pytorch Predictions & Scores:** </font>",
                unsafe_allow_html=True,
            )
            st.write(pd.DataFrame.from_dict(label_string))
            
        with x3:
            st.markdown(
                "<font size='4'> **Keras Predictions & Scores:** </font>",
                unsafe_allow_html=True,
            )
            st.write(pd.DataFrame.from_dict(predicted_string_data))
            
        c1, c2, c3 = st.columns((2, 2, 2))
        
        with c1:
            st.markdown(
                "<font size='4'> **TruLens Attention Mask:** </font>",
                unsafe_allow_html=True,
            )
            # Processing the data
            infl = InputAttribution(model)
            attrs_input = infl.attributions(x_pp)
            masked_image = MaskVisualizer(blur=10, threshold=0.95)(attrs_input, x, output_file="mask.jpg")
            
            # Removing Masked Visualizer Border
            img = PIL.Image.open("mask.jpg")
            bg1 = PIL.Image.new(img.mode, img.size, img.getpixel((0,0)))
            diff1 = PIL.ImageChops.difference(img, bg1)
            diff1 = PIL.ImageChops.add(diff1, diff1, 2.0, -100)
            bbox1 = diff1.getbbox()
            if bbox1:
                img = img.crop(bbox1)
            
            # Display Masked Image
            st.image(img.resize((256, 256), PIL.Image.LANCZOS))
        with c2:
            st.markdown(
                "<font size='4'> **TruLens Attention Heatmap:** </font>",
                unsafe_allow_html=True,
            )
            # Setup Heatmap Visualizer
            infl = IntegratedGradients(model, resolution=10)
            attrs_input = infl.attributions(x_pp)
            masked_image = HeatmapVisualizer(blur=10, )(attrs_input, x, output_file="heatmap.jpg", fig=True)
            im = PIL.Image.open("heatmap.jpg")
            
            # Removing Heatmap Border
            bg = PIL.Image.new(im.mode, im.size, im.getpixel((0,0)))
            diff = PIL.ImageChops.difference(im, bg)
            diff = PIL.ImageChops.add(diff, diff, 2.0, -100)
            bbox = diff.getbbox()
            if bbox:
                im = im.crop(bbox)
                
            # Display Heatmap
            st.image(im.resize((256, 256), PIL.Image.LANCZOS))
        with c3:
            st.markdown(
                "<font size='4'> **Keras Attention Heatmap:** </font>",
                unsafe_allow_html=True,
            )
            
            st.image(heatmap.resize((256, 256), PIL.Image.LANCZOS))

        # Define the influence measure.
        infl = InternalInfluence(model, 'features_28', 'max', 'point')

        # Get the attributions for the internal neurons at layer 'features_28'. Because 
        # layer 'features_28' contains 2D feature maps, we take the sum over the width 
        # and height of the feature maps to obtain a single attribution for each feature 
        # map.
        attrs_internal = infl.attributions(x_pp).sum(axis=(2,3))

        # Value represents some type of learned feature that was the most important in the 
        # networks' decision to label this point as beagle
        top_feature_map = int(attrs_internal[0].argmax())

        st.write('Top feature map:',top_feature_map)
        masked_image = ChannelMaskVisualizer(
            model,
            'features_28',
            top_feature_map,
            blur=10,
            threshold=0.95)(x, x_pp)
        plt.axis('off')
        st.image(masked_image[0].transpose((1,2,0)), caption="Top Feature Activation Map")
        

if __name__ == "__main__":
    main()

