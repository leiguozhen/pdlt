from utility import *

# faces_config


output_dir = './my_faces/'
detector = get_detector()
camera = get_camera()
make_dir(output_dir)
faces_counter = 0

while faces_counter < MAXFACES:
    print('Now we have ', faces_counter, ' faces...')
    faces = get_faces_from_camera(camera, detector, SIZE)
    for face in faces:
        faces_counter += 1
        show_face(face)
        save_face(face, output_dir, faces_counter)

    # 检测ESC按键退出
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break
