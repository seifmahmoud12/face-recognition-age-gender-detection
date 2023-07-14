import cv2
import itertools
import os
import pymysql
# import pymysql as pymysql
#
# # database connection
# connection = pymysql.connect(host="127.0.0.1", port=3306, user="root", passwd="", database="attend",    autocommit=True)
# cursor = connection.cursor()
# import csv
# from datetime import datetime
# import pandas as pd
# now = datetime.now()
# #datestring = now.strftime("%Y-%m-%d")
# empdata = pd.read_csv('1.csv', index_col=False, delimiter = ',')
#
#
# #sheet = book.sheet_by_name("attendnancedeparture")
#
#
# for i, row in empdata.iterrows():
#             # here %S means string values
#             query = """INSERT INTO empid (empID,arrivaltime,date) VALUES (%s, %s,%s)"""
#             # if tuple(row(2:))
#             cursor.execute(query, tuple(row))
#             print("Record inserted")
#
#             #cursor.commit()
#
#
#
# connection.close()
dataset_folder = "data/"
cap = cv2.VideoCapture(0)
#

my_name = "30"
os.mkdir(dataset_folder + my_name)
num_sample =3000

i = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imshow("Capture Photo", frame)
        cv2.imwrite("data/%s/%s_%04d.jpg" % (my_name, my_name, i), cv2.resize(frame, (250, 250)))

        if cv2.waitKey(100) == ord('q') or i == num_sample:
            break
        i += 1
cap.release()
cv2.destroyAllWindows()