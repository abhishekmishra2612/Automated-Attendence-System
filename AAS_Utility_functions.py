import cv2
import os
import face_recognition as fr
from datetime import datetime


image_path = 'Images'
image_files = os.listdir(image_path+'/')

# Load image from directory and return face Encodings & their Roll No.
def get_Encode_Roll(file_name):
    image_path = "Images/"
    image = cv2.imread(image_path+file_name)
    roll = file_name.split('.')[0]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (0, 0), None, 0.25, 0.25)
    return fr.face_encodings(image)[0], roll

known_face_encodings, Rolls = [], []
for file_name in image_files:
    encode, roll = get_Encode_Roll(file_name)
    known_face_encodings.append(encode)
    Rolls.append(roll)


roll_to_name ={'160202':'Abhishek Mishra', '160217':'Deepak S Mudila', '160221':'Dheeraj Kapri', '160249':'Shivam Maindola', '160277':'Larry Page', '160275':'Sushant S Rajput', '160281':'Barak Obama','160276':'Elon Musk', '160278':'Narendra Modi', '160279':'Jeff Bezos'}


# This fuction is use to mark the attendence of a student whose attendence did not marked.
def mark_Attendence(roll):
    with open("Attendence_file.txt", "r+") as f:
        dataList = f.readlines()
        rollList = []
        for line in dataList:
            entry = line.split(',')
            rollList.append(entry[0])
        if roll not in rollList:
            now = datetime.now() # Time of arrival
            DTstring = now.strftime("%H:%M:%S")
            f.write(f'\n{roll}, {roll_to_name[roll]}, {DTstring}')
