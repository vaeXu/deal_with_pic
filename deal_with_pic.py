import numpy as np
import cv2
import os.path
import sys
import os
def one_pic(image):
    print('read a image.....')
    image = cv2.imread(image)

    # 先切出人脸后进行模糊度判断
    face = detectFace(image)
    print('detected image.......')
    cv2.imshow('1', face)
    # show image
    cv2.imwrite('/home/xjh/Desktop/face/1.jpg', face)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
def detectFace(image):
    face_cascade = cv2.CascadeClassifier(r'/home/xjh/xu_project/Image_deal/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image)
    # 只考虑一个人的情况
    if len(faces) > 0:
        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]

        face = image[y:y + h, x:x + w]

        return face
    else:
        pass
def video_pic(video_path, save_path):
    # path = '/home/xjh/Desktop/face/false/'

    print('read a video ...')
    cap = cv2.VideoCapture(video_path)
    i = 0
    flag = 0
    while True:

        ret, frame = cap.read()
        if ret == True:
            if flag % 3 == 0 :
                # print('检测并切割第{}张人脸......'.format(str(i)))
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = detectFace(frame)
                image_var = just_pic(face)
                if image_var == None:
                    pass
                else:
                    if i < 300:
                        # path = save_path + str(i) + '_' + str(image_var) + '.jpg'
                        path = save_path + str(i) + '.jpg'
                        print(path)
                        cv2.imwrite(path, face)
                    else:
                        break

                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            flag += 1
            i += 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def just_pic(face):

    # 返回图像的模糊度
    if face is not None:
        # 中值降噪q
        res = cv2.medianBlur(face, 5)

        # 返回模糊度
        imageVar = cv2.Laplacian(res, cv2.CV_64F).var()
        # print('.......................................')
        # 如果模糊度大于85则存起来
        x = np.float64(0)
        if imageVar > x:
            print('模糊度 == ', imageVar)
            return imageVar
        else:
            pass

            # address 这里要有.jpg .png等图片后缀

            # 格式: 名字_模糊度_.jpg
            # eg: ts1_86.8853801969.jpg
            # str(imageVar)

        #     address = '/home/xjh/Desktop/face/' + '1' + '.jpg'
        #     cv2.imwrite(address, face)
        # else:
        #     pass
def open_file_dir(path):
    j = 0
    for i in os.walk(path):
        if j > 0:
            # print(i[0])
            for k in os.listdir(i[0]):
                # k 表示文件夹下面的所有文件
                path = str(i[0]) + '/' + k
                print(path)
                save_path = str(i[0]) + '/' + 'F'
                if os.path.exists(save_path):
                    remove(save_path)
                    os.makedirs(save_path)
                    print('..........................')
                    video_pic(path, save_path)

                else:
                    print('..............////////////////////...............', save_path)
                    os.makedirs(save_path)
                    video_pic(path, save_path)

        j += 1


def save_face(path, face):
    pass

def remove(path):
    # 删除空文件夹或者空文件
    for file in os.listdir(path):
        print(file)
        file = path + file
        if os.path.isdir(file):
            if not os.listdir(file):
                os.rmdir(file)
        elif os.path.isfile(file):
            if os.path.getsize(file == 0):
                os.remove(file)
def just_size(path):
    files = os.listdir(path)
    for file in files:
        path_ = os.path.join(path, file)
        # 把小于3K的图像都删除掉，0-3之间属于误切误认范围
        if os.path.getsize(path_) <6000:
            os.remove(path_)
            print(path_)
        else:
            pass



if __name__ == '__main__':
    just_size('/home/xjh/Desktop/jhd_false_test/')
    # image = '/home/xjh/Pictures/star_1.jpg'
    # one_pic(image)
    # print('...................................')
     # video_pic(video_path='/home/xjh/Downloads/train_release/real/HR_4(副本).avi', save_path='/home/xjh/Desktop/face/reals/')
    # print('*************')

    # base_path = '/home/xjh/Downloads/train_release/real/'
#    base_path = '/home/xjh/Downloads/replay_attack/test/real/'
    #save_path = '/home/xjh/Downloads/replay_attack/test/real_jpg/'
    #j = 0
    #k = 0
    #for i in os.listdir(base_path):
        # print(i)
       # print('正在处理视频 __{}'.format(i))
       # video_path = base_path + i
       # save_path_ = save_path + str(j) + '_'

        #
        # print('.....................', save_path_)
    #    # # pdb.set_trace()
    #    video_pic(video_path, save_path_)
    #    j += 1
        # print('finsh {}'.format(str(i[0])))




    # open_file_dir('/home/xjh/Downloads/train_release/false')
    # remove('/home/xjh/Downloads/train_release/false/')




