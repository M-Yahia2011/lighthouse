"""
Face Comparing Model

Original file is located at
    https://colab.research.google.com/drive/1FJYzj2ZbdzAUBvL5r4RxqtWrqkrK-n6j
    
FaceNet Model Link at
    https://drive.google.com/file/d/1EpKBNcQ5rwVAuEXaMIk_F4PngMuvgzTD/view?usp=sharing
"""

# !sudo pip install mtcnn

import mtcnn

"""# Import libs"""
from os import listdir
from PIL import Image
from numpy import asarray, dot,expand_dims
from numpy.linalg import norm
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN #Multitask Cascaded CNN
from tensorflow.keras.models import load_model
from numpy.linalg import norm

"""# Extract_face """
# create the detector, using default or trained weights
detector = MTCNN()
### we create it outside of the function to prevent
### retracing and recreating the object
def extract_face(filename, required_size=(160, 160)):
    '''This function takes an image and extracts all faces from it'''
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # >>>>>>>>>>>>>>>>>>>>><-M.Y.E->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ###########Wrong place to create an instance###########  detector = MTCNN()
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # detect faces in the image 
    results = detector.detect_faces(pixels)
    
    # extract the bounding box of all faces in the image
    faces_bound_boxes = [result['box'] for result in results]
    face_list = []
    for i in range(len(faces_bound_boxes)):
        x1, y1, width, height = faces_bound_boxes[i]
        # negative values bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        face_list.append(face_array)
    return face_list

"""
# Testing extract_face function
#### Notice the detected ghosts in the output
"""
def test_extract_face(img):
    # show original image
    img = Image.open(img)
    pyplot.axis('off')
    pyplot.title('original')
    pyplot.imshow(img.convert('RGB')) # without the conversion the colors are strange
    pyplot.figure()
    pyplot.show()
    ls = extract_face(img)
    for face in ls:
        pyplot.axis('off')
        pyplot.imshow(face)
        pyplot.figure()
        pyplot.show()

"""# Import face-net pretrained model from google drive"""
# from google.colab import drive
# drive.mount('/content/gdrive')
# dirve_PATH = 'gdrive/MyDrive/Colab Notebooks/facenet_keras.h5'

facenet = load_model('./facenet_keras.h5', compile=False)
# print(facenet.summary())

def prepare_face(face_pixels):
    '''To predict an embedding, first the pixel values of the image need to be suitably prepared to meet the expectations of the FaceNet model.
       This specific implementation of the FaceNet model expects that the pixel values are standardized'''

    face_pixels = face_pixels.astype('float32')
    mean , std = face_pixels.mean() , face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    #In order to make a prediction for one example in Keras, we must expand the dimensions so that the face array is one sample.
    sample = expand_dims(face_pixels, axis=0)
    return sample

def get_embedding(face):
    """return the embedding of a face"""
    return facenet.predict(face)[0]

def get_similarity(face1, face2):
    cos_sim = dot(face1, face2)/(norm(face1)*norm(face2))
    return cos_sim

def wrapper(image1, image2):
    """This function uses all of the previous functions to give the similarity of two input faces"""
    faces_in_image1 = extract_face(image1)
    faces_in_image2 = extract_face(image2)
    '''
    # <<-M.Y.E->>
    think we need to specify image1 and image2 to determin which is I/P or O/P
    this is for the purpose of using MTCNN once for the input as we should have saved the face embedding
    in the Database
    '''
    """
    if (len(faces_in_image1) > 1) or (len(faces_in_image) > 1):
        print("More Than 1 Face is not allowed")
        return None
    """
    '''
    # <<-M.Y.E->> 
    # check if there is more than 1 face in either of them 
    we can speed this up a bit by stop the face appending early in the function [extract_face()]
    as we wait for the entire appending to finish and return through the call we made of the fn 
    which cause unnecessary delay :) :D  
    ### but there are ghost images that the MTCNN generates.
    this will stop the comparison if we stop the wrapper here 
    POSSIBLE SOLUTIONS
    # is appending just 2 will sort this out ??? --> could work if the right face is the first element in the list
    # we can use a model for Face Verification but there is enough models already !!
     
    '''

    l1, l2 = [], []
    for face in faces_in_image1:
        face = prepare_face(face)
        emb = get_embedding(face)
        l1.append({'face':face , 'embedding':asarray(emb)})

    for face in faces_in_image2:
        face = prepare_face(face)
        emb = get_embedding(face)
        l2.append({'face':face , 'embedding':asarray(emb)})   
    
    print("No. of faces in each images",len(l1),":",len(l2))
    
    # >>>>>>>>>>
    # original faces
    ls = [image1, image2]
    for img in ls:
        img = Image.open(img)
        pyplot.axis('off')
        pyplot.title('original')
        pyplot.imshow(img.convert('RGB')) # without the conversion the colors are strange
        pyplot.figure()
        pyplot.show()
    # >>>>>>>>>>

    for face1 in l1:
        for face2 in l2:
            sim = get_similarity(face1['embedding'] , face2["embedding"])

            face1_1 = face1['face'].reshape(160,160,3)
            face2_2 = face2['face'].reshape(160,160,3)
            print('The first extracted face is ')
            pyplot.imshow(face1_1)
            pyplot.show()
            print('The second extracted face is ')
            pyplot.imshow(face2_2)
            pyplot.show()
            print('and the similarity between them is {}'.format(sim))
            print('##################################################################################################################')

# TESTING
# test_extract_face('/content/tesla_ghosts.png')
wrapper('/content/m4.jpg' , '/content/m6.jpg')
