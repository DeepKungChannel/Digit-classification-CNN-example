import tensorflow as tf
import numpy as np
import gradio as gr



model = tf.keras.models.load_model('mnist_model_cnn.h5')

def classify(img):
    global model
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1,28,28,1)
    prediction = model.predict(img).tolist()
    return {str(i) : prediction[0][i] for i in range(10)}

def main():
    inputs = gr.inputs.Image(image_mode='L', 
                    source='canvas', 
                    shape=(28, 28), 
                    invert_colors=True, 
                    tool= 'select')
    output = gr.outputs.Label(num_top_classes=10)
    gr.Interface(classify, inputs, output,title="Mnist prediction").launch()


if __name__ == '__main__':
    main()