# Untitled - By: cyrilmonette - jeu. juin 16 2022

import sensor, image, time, os, tf

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

#net = tf.load('lite-model_ssd_mobilenet_v2_fpn_100_uint8_default_1.tflite', load_to_fb=True) # load_to_fb=True means Model stored on OpenMV Cam frame buffer stack (32MB)

print("Loading network")
#net = tf.load("person_detection")
#net = tf.load("lite-model_aiy_vision_classifier_birds_V1_3.tflite",load_to_fb=True)
print("network loaded")

labels = []

try: # Load labels if they exist
    labels = [line.rstrip('\n') for line in open("labels_mobilenet_quant_v1_224.txt")]
    print("Labels read")
except:
    pass
    print("Could not read labels")

colors = [ # Colors used to draw the rectangles around the detected objects. Add more colors if you are detecting more than 7 types of classes at once.
    (255,   0,   0),
    (  0, 255,   0),
    (255, 255,   0),
    (  0,   0, 255),
    (255,   0, 255),
    (  0, 255, 255),
    (255, 255, 255),
]

clock = time.clock()
while(True):
    print("loop")
    clock.tick()

    img = sensor.snapshot()
    #normalized_input_image_tensor: an uint8 array of shape [batch, height, width, channels]
    #where batch = 1, channels = 3, and height and width can be any size.
    #Values should be between [0, 255]

    classification=tf.classify("lite-model_aiy_vision_classifier_birds_V1_3.tflite",img)


    #result = zip(labels, classification.classification_output())
    #print(classification.classification_output())


    print(clock.fps(), "fps", end="\n\n")
