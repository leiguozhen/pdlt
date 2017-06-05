from utility import *


output = cnnLayer()
predict = tf.argmax(output, 1)
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('.'))


def is_my_face(image):
    res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})
    return True if res[0] == 1 else False

camera = get_camera()
detector = get_detector()

while True:
    _, img = camera.read()

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)

    if not len(dets):
        print('don`t have faces')
        continue

    for i, d in enumerate(dets):
        x1 = max(0, d.top())
        y1 = max(0, d.bottom())
        x2 = max(0, d.left())
        y2 = max(0, d.right())
        face = img[x1:y1, x2:y2]
        face = cv2.resize(face, (SIZE, SIZE))
        print('Is this my face? %s' % is_my_face(face))

        cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
        cv2.imshow('image', img)

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)

sess.close()
