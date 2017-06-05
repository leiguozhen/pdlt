
from utility import *


my_faces_path = './my_faces'
other_faces_path = './other_faces'
MAXROUND = 10
imgs = []
labs = []


def readData(imgs, labs, path, h=SIZE, w=SIZE):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            top, bottom, left, right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分，成黑边
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))
            imgs.append(img)
            labs.append(path)
    return imgs, labs

readData(imgs, labs, my_faces_path)
readData(imgs, labs, other_faces_path)

# 将图片数据与标签转换成numpy数组
imgs = np.array(imgs)# 数组里面的对象仍然是img
labs = np.array([[0, 1] if lab == my_faces_path else [1, 0] for lab in labs])

# 随机划分测试集与训练集
train_x, test_x, train_y, test_y = \
    train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0, 100))

# 参数：图片数据的总数，图片的高、宽、通道
train_x = train_x.reshape(train_x.shape[0], SIZE, SIZE, 3) # 数组维度重新划分
test_x = test_x.reshape(test_x.shape[0], SIZE, SIZE, 3)

# 将数据转换成小于1的数
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0


# 图片块，每次取100张图片
batch_size = 100
num_batch = len(train_x) // batch_size


def cnnTrain():
    out = cnnLayer()
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))
    train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))

    # 将loss与accuracy保存以供 tensor-board 使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()


    # 数据保存器的初始化
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())


    alltimes = 0
    for round in range(MAXROUND):
         # 每次取128(batch_size)张图片, 一共取num_batch次
        for cur_batch in range(num_batch):
            batch_x = train_x[cur_batch*batch_size: (cur_batch+1)*batch_size]
            batch_y = train_y[cur_batch*batch_size: (cur_batch+1)*batch_size]
            # 开始训练数据，同时训练三个变量，返回三个数据
            _, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                        feed_dict={x: batch_x, y_: batch_y, keep_prob_5: 0.5, keep_prob_75: 0.75})

            alltimes = round * num_batch + cur_batch
            summary_writer.add_summary(summary, alltimes)

            # debug: 打印损失
            print(alltimes, loss)

    acc = accuracy.eval({x: test_x, y_: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0}, session=sess)
    print(acc)
    saver.save(sess, './train_faces.model', global_step=alltimes)
    sess.close()


cnnTrain()
