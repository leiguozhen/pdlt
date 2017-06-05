
from utility import *

# faces-geter-config
input_dir = './input_img'
output_dir = './other_faces'

make_dir(output_dir)
detector = get_detector()

faces_counter = 0
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        print('#Now we have ', faces_counter, ' faces ...')
        if filename.endswith('.jpg'):
            faces = get_faces_from_picture(detector, path+'/'+filename, SIZE)
            for face in faces:
                faces_counter += 1
                save_face(face, output_dir, faces_counter)


