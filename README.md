# Smart-Attendance

The program above marks the attendance of all the stuents in a class given a 15 second video of the class (20-30 students).
Uses technologies like YOLOv5 for face detection and recognition. 
Generates vectors representing facial features of enrolled students and compares them to students present in the class.
Change parameter of face_location in multithreaded.py from model = cnn to model = hog to run without cuda. model = hog is extremely slow on CPUs
