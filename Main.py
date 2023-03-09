import scipy.io.wavfile as wav
from tkinter import *
from tkinter import filedialog
import tkinter.messagebox #弹窗库
import cv2
import numpy as np
import tkinter.font as tkFont
import tkinter.filedialog
from tkinter.simpledialog import askstring
from tkinter.ttk import *

np.random.seed(2022)
# 生成随机数种子
stop_mark = np.random.randint(0, 2, 128)
# 返回128个整数，要么是0要么是1，实际上可以认为生成了一串长为128的随机比特流
# stop_mark的用处：
def plus(str):
    return str.zfill(8)


# Python zfill() 方法返回指定长度的字符串，原字符串右对齐，前面填充0。

def get_key(strr):
    # 获取要隐藏的文件内容
    tmp = strr
    f = open(tmp, "rb")
    str = ""
    s = f.read()
    global text_len
    text_len = len(s)
    for i in range(len(s)):
        # code.interact(local=locals())
        str = str + plus(bin(s[i]).replace('0b', ''))  # 此处获得的ASCII码完全正确
    # 逐个字节将要隐藏的文件内容转换为二进制，并拼接起来
    # 1.先用ord()函数将s的内容逐个转换为ascii码
    # 2.使用bin()函数将十进制的ascii码转换为二进制
    # 3.由于bin()函数转换二进制后，二进制字符串的前面会有"0b"来表示这个字符串是二进制形式，所以用replace()替换为空
    # 4.又由于ascii码转换二进制后是七位，而正常情况下每个字符由8位二进制组成，所以使用自定义函数plus将其填充为8位
    # print str
    f.closed
    return str


def mod(x, y):
    return x % y;

def toasc(strr):
    return int(strr, 2)

class LSBEmbedder:
    def __init__(self, seed, rate=0.9, mode='single'):
        self.rate = rate # 隐写率
        self.seed = seed # 随机数种子
        self.mode = mode # single:对单个wav进行处理 match:对文件夹下多个wav进行操作
        self.stop_mark = stop_mark
        self.channels = None # channels为1：单通道WAV，channels为2：双通道WAV
        self.fs = None # fs：采样率
        self.left_signal = [] # 左声道
        self.right_signal = [] # 右声道
        self.wavsignal = [] # 数据块
        self.embed_length = 0

    def _waveReader(self, path):
        """
        read wave file and its corresponding
        :param path: wav fi
        :return:
        """
        fs, wavsignal = wav.read(path) # 采样率为fs，数据为wavsignal
        # wavsignal是一个numpy数组
        # 对于 1 通道 WAV，数据是 1-D，否则是 2-D 形状(Nsamples，Nchannels)。
        # 如果传递了没有 C-like 文件说明符(例如 io.BytesIO )的 file-like 输入，则这将不可写。
        # 就目前的代码而言，并不能正确读取双通道WAV文件，怎么会使呢？
        self.fs = fs
        print(wavsignal.ndim)
        if wavsignal.ndim == 2: # len函数返回numpy数组的长度，而非元素个数
            # 例如：wavsignal=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]，则len(wavsignal) == 3
            # 因此，wavsignal的形态有且只有两种：
            # [1,2,3,...,x] 或 [[1,2,3,...,x],[a,b,c,...,x]]，第二种显然为 2 通道的WAV
            # 上面一行是代码原作者的理解，实际上双通道的WAV在读入后并不是这个形式
            # 应当为[[1,2],[3,4],[5,6],[x,x],...]形式
            self.channels = 2
            self.wavsignal = wavsignal
            for slice in wavsignal:
                self.left_signal.append(slice[0])
                self.right_signal.append(slice[1])
        else:
            self.channels = 1
            self.wavsignal = wavsignal

    def _LSBReplace(self, wavsignal, secret_message):
        """
        embed watermarking a single wave
        :param secret_message: secret message
        :return:
        """
        # choose random embedding location with roulette
        np.random.seed(self.seed)
        roulette = np.random.rand(len(wavsignal))
        # 通过此函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
        # 此roulette数组的长度和wavsignal一样，可以认为：每一个二进制信号信息都对应一个roulette的值
        stego = np.array(wavsignal)
        k = 0
        for i in range(len(wavsignal)):
            if roulette[i] <= self.rate:
                # 如果信息i对应的roulette值小于等于rate，就执行message操作
                # 猜测roulette操作是为了将secret_message 随机 加入到原有信息中
                # 由于rate设置为0.9，所以0.9乘以原有信息长度需要大于secret_message信息长度
                # 这部分解释了为什么有以下assert:
                # assert self.channels * len(self.wavsignal) * self.rate >= 1.1 * len(secret_message)
                # 但是1.1和self.channels的意义尚不明确
                value = wavsignal[i]
                # 对于单声道wav，则value为单值，对于双声道wav，value为存有两个值的数组（左右声道）使用batch mode
                # print(value)
                if value < 0:
                    value = -value
                    # 若小于0，则先利用python对于正负整数的转换机制将其变为正数int，再进行二进制串操作
                    negative = True
                else:
                    negative = False
                # embed secret bit
                if k < len(secret_message) and int(secret_message[k]) == 0:
                    value = value & 0b1111111111111110
                    k += 1
                elif k < len(secret_message) and int(secret_message[k]) == 1:
                    value = value | 0b0000000000000001
                    k += 1
                    # 用16位的二进制流去做水印，如果secret_message的k位数字是1，则将原有信息的此位末尾or 1
                    # 否则，将原有信息的此位末尾and 0
                    # 实际上这里就是LSB算法的体现，只修改最低有效位而对其他位数不做任何修改
                if negative:
                    stego[i] = -value
                    # 了转反
                else:
                    stego[i] = value
                    # 没反转

        stego = np.array(stego).astype(np.int16)
        return stego

    def _saveWave(self, stego, cover_path, stego_path, inplace=False):
        """
        save stego wave
        :param stego: stego wavsignal
        :param cover_path: cover path
        :param stego_path: stego path
        :param inplace: whether to save in cover path
        :return:
        """
        if inplace:
            wav.write(cover_path, self.fs, stego)
        else:
            assert stego_path is not None
            wav.write(stego_path, self.fs, stego)

    def embed(self, cover_path, stego_path, secret_message, inplace=False):
        """
        steganogaphy
        :param cover_path: cover wave path
        :param stego_path: stego wave path
        :param secret_message: secret message
        :param inplace: steganogarphy in place
        :return:
        """
        # add stop mark
        secret_message = np.concatenate([secret_message, self.stop_mark], axis=0)
        # 这里是做一个数组的拼接，把
        # https://www.cnblogs.com/shueixue/p/10953699.html
        # pre check
        self._waveReader(cover_path)
        # print("channels:"+str(self.channels))
        # print("self.wavsignal.shape.size:"+str(len(self.wavsignal)))
        # print("self.rate:"+str(self.rate))
        # print("1.1*len(secret_message):"+str(len(secret_message)))
        # print(self.channels * len(self.wavsignal) * self.rate)
        # print(1.1*len(secret_message))
        assert self.channels * len(self.wavsignal) * self.rate >= 1.1 * len(secret_message)
        # 上：如果当前处理的音频，声道数 * 采样数 * 采样率（？） 大于等于 1.1*密文长度，则成立
        assert self.channels in [1, 2] # 如果不在1，2就弹出，说明不为单双通道，但是好像纯纯没用

        # embed secret message
        if self.channels == 1:
            if self.mode == 'single':
                stego = self._LSBReplace(self.wavsignal, secret_message) # stego 为ndarray类型的变量
                self._saveWave(stego, cover_path, stego_path, inplace)
            elif self.mode == 'batch':
                for i in range(len(cover_path)):
                    stego = self._LSBReplace(self.wavsignal, secret_message)
                    self._saveWave(stego, cover_path, stego_path, inplace)
        elif self.channels == 2:
            if self.mode == 'single':
                left_stego = self._LSBReplace(self.left_signal, secret_message)
                right_stego = self._LSBReplace(self.right_signal, secret_message) # 二者属性都是nparray
                stego = np.vstack([left_stego, right_stego])
                stego = stego.T
                self._saveWave(stego, cover_path, stego_path, inplace)
            elif self.mode == 'batch':
                # the same secret messages are embedding in different carrier
                for i in range(len(stego_path)):
                    left_stego = self._LSBReplace(self.left_signal, secret_message)
                    right_stego = self._LSBReplace(self.right_signal, secret_message)
                    # stego = [left_stego, right_stego]
                    stego = np.vstack([left_stego, right_stego])
                    stego = stego.T
                    self._saveWave(stego, cover_path[i], stego_path[i], inplace)

class LSBExtractor:
    def __init__(self, seed, rate=0.9):
        self.seed = seed
        self.stop_mark = stop_mark
        self.rate = rate
        self.fs = None
        self.channels= None
        self.wavsignal = []
        self.left_signal = []
        self.right_signal = []

    def _waveReader(self, path):
        """
        read wave file and its corresponding
        :param path: wav fi
        :return:
        """
        fs, wavsignal = wav.read(path)
        self.fs = fs
        if wavsignal.ndim == 2: # 根据WAV文件的单通道双通道属性，将其保存到self中
            self.channels = 2
            self.wavsignal = wavsignal
            for slice in wavsignal:
                self.left_signal.append(slice[0])
                self.right_signal.append(slice[1])
        else:
            self.channels = 1
            self.wavsignal = wavsignal

    def _checkHeader(self, header):
        """
        check the validness of header
        :param header: header
        :return: True/False
        """
        return True

    def _checkStop(self, message):
        """
        check stop
        :param message: secret message
        :return: True/False
        """
        # print("self.stop_mark:"+str(self.stop_mark))
        message_stop = message[-len(self.stop_mark):]
        # 从已经添加了stop_mark的message中取回stop_mark
        count = 0
        for i in range(len(self.stop_mark)):
            if message_stop[i] == self.stop_mark[i]:
                count += 1

        if count == len(self.stop_mark):
            return True
        else:
            return False
        # 检查从message中取得的stop_mark是否完整

    def _LSBExtract(self, roulette, wavsignal):
        """
        extract LSB from stego wavsignal
        :param roulette:
        :param wavsignal:
        :return: secret message
        """
        message = []
        for i in range(len(wavsignal)):
            if roulette[i] <= self.rate:
                value = wavsignal[i]
                value = '{:016b}'.format(value)
                message.append(int(value[-1]))
                # 读取嵌入信息，如若该位置的roulette值小于rate，说明被修改了，因此需要调用format还原

                # check the validness of header
                if len(message) == 44:
                    assert self._checkHeader(message) is True

                # check stop mark
                if len(message) >= len(self.stop_mark) and self._checkStop(message):
                    return message
        return message

    def extract(self, wave_path, message_path):
        """
        extract message in wave
        从已有的wav文件里提取信息
        :param wave_path: wave path
        :param message_path:  message path
        :return:
        """
        # choose random embedding location with roulette
        # 随机选择嵌入位置
        self._waveReader(wave_path)
        np.random.seed(self.seed)
        roulette = np.random.rand(len(self.wavsignal))
        if self.channels == 1:
            message = self._LSBExtract(roulette, self.wavsignal)
        elif self.channels == 2:
            message_left = self._LSBExtract(roulette, self.left_signal)
            message_right = self._LSBExtract(roulette, self.right_signal)
            message = np.vstack([message_left, message_right])

        messageint = ""
        for i in message:
            if i == 0:
                messageint+="0"
            else:
                messageint+="1"
        if message_path == "":
            return message
        else:
            with open(message_path, "wb") as f:
                for i in range(0, len(messageint) - 128, 8):
                    stra = toasc(messageint[i:i + 8])
                    stra = chr(stra)
                    sb = bytes(stra, encoding="utf8")
                    f.write(sb)
                    stra = ""
            f.closed
        return message

def ChooseWAV():
    tkinter.messagebox.showinfo('提示', '请选择WAV音频文件')
    Fpath = filedialog.askopenfilename()
    return Fpath

def ChooseMP4():
    tkinter.messagebox.showinfo('提示', '请选择MP4视频文件')
    Fpath = filedialog.askopenfilename()
    return Fpath

def ChooseWAVName():
    tkinter.messagebox.showinfo('提示', '请输入WAV文件名（无需加后缀）')
    Fpath = askstring("请输入一个字符或字符串",prompt="文件名：") + ".wav"
    return Fpath

def ChoosePath():
    tkinter.messagebox.showinfo('提示', '请选择保存路径')
    IMG_PATH = filedialog.askdirectory()
    return IMG_PATH

def GetKeylenMP4():
    tkinter.messagebox.showinfo('提示', '请设置提取信息长度')
    keylen = askstring("请输入需要提取的信息长度（阿拉伯数字），长度为",prompt="：")
    keylen = int(keylen)*8
    return keylen

def GetMessagePath():
    tkinter.messagebox.showinfo('提示', '请选择需要嵌入的txt文件')
    Fpath = filedialog.askopenfilename()
    return Fpath

def BGR2RGB(im_bgr):
    im_rgb = im_bgr[:, :, ::-1]
    # Image.fromarray(im_rgb).save(r'C:\Users\86180\Desktop\example-trans.jpg')
    return im_rgb

def RGB2BGR(im_rgb):
    im_bgr = im_rgb[:, :, ::-1]
    return im_bgr

def GetFrameFromVideo():
    address = ChooseMP4()
    if address == "":
        tkinter.messagebox.showinfo("提示",'请选择一个有效的mp4文件')
        return

    video = cv2.VideoCapture(address)  # 读取视频
    fps = video.get(cv2.CAP_PROP_FPS)
    frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    secretmessage = GetMessagePath()
    if secretmessage == "":
        tkinter.messagebox.showinfo("提示", '请选择一个有效的txt文件')
        return

    stego_path = ChoosePath()
    if stego_path == "":  # 啥也没选，退出吧！
        tkinter.messagebox.showinfo('提示', '请选择一个有效的视频保存路径！')
        return

    success, frame = video.read()  # success 是bool类型，其值代表是否读取帧成功，frame 是读取的帧文件，以BGR格式存储
    index = 0

    videoWriter = cv2.VideoWriter(stego_path+'\Trans.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps,
                                  size)

    key = get_key(secretmessage)
    keylen = len(key)

    while success:
        if index < keylen:
            cope_with_frame = BGR2RGB(frame)
            if key[index] == '0':
                cope_with_frame[1, 1] = np.array([0, 0, 250])
            else:
                cope_with_frame[1,1] = np.array([250,0,0])
            frame = RGB2BGR(cope_with_frame)
        videoWriter.write(frame)
        success, frame = video.read()
        index += 1

    videoWriter.release()
    video.release()

def GetInfoFromVideo():
    address = ChooseMP4()
    if address == "":
        tkinter.messagebox.showinfo("提示", '请选择一个有效的mp4文件')
        return

    path = ChoosePath() + '\RecoverFromVideo.txt'
    video = cv2.VideoCapture(address)  # 读取视频
    fps = video.get(cv2.CAP_PROP_FPS)
    recover = ""
    frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    success, frame = video.read()  # success 是bool类型，其值代表是否读取帧成功，frame 是读取的帧文件，以BGR格式存储
    index = 0
    keylen = GetKeylenMP4()

    while success:
        if index < keylen:
            cope_with_frame = BGR2RGB(frame)
            pixel = cope_with_frame[1,1]
            print(pixel)
            if 3*pixel[0] < pixel[2]:
                recover+="0"
            else:
                recover+="1"
            frame = RGB2BGR(cope_with_frame)
        success, frame = video.read()
        index += 1

    with open(path, "wb") as f:
        for i in range(0, len(recover), 8):
            stra = toasc(recover[i:i + 8])
            stra = chr(stra)
            sb = bytes(stra, encoding="utf8")
            f.write(sb)
            stra = ""
    f.closed
    video.release()

def DoWaveLSB():
    wave_path = ChooseWAV()
    if wave_path == "": # 啥也没选，直接退出吧！
        tkinter.messagebox.showinfo('提示', '请选择一个有效的源文件路径！')
        return

    stego_path = ChoosePath()
    if stego_path == "": # 啥也没选，退出吧！
        tkinter.messagebox.showinfo('提示', '请选择一个有效的WAV保存路径！')
        return

    wavname = ChooseWAVName() # 啥也没选，退出吧！
    if wavname == ".wav":
        tkinter.messagebox.showinfo('提示', '请输入一个有效的文件名！')
        return

    stego_path = stego_path + "/" + wavname

    message_path = GetMessagePath()
    if message_path == "":
        tkinter.messagebox.showinfo('提示', '请选择一个有效的txt文件进行嵌入！')
        return

    key = get_key(message_path)
    key = list(key)
    secret_message = np.array(key)
    # 生成一个长度为1600的随机01二进制串，即为水印信息
    alice = LSBEmbedder(0, mode='single')
    alice.embed(wave_path, stego_path, secret_message)

    bob = LSBExtractor(0)
    m = bob.extract(stego_path, "")

    count = 0

    if bob.channels == 1:
        assert len(m) - len(stop_mark) == len(secret_message)
        for i in range(len(secret_message)):
            if int(m[i]) != int(secret_message[i]):
                count += 1
        print('BER:', count / len(m))
    else:
        assert len(m[0]) - len(stop_mark) == len(secret_message)
        assert len(m[1]) - len(stop_mark) == len(secret_message)
        m1 = m[0]
        m2 = m[1]
        for i in range(len(secret_message)):
            if int(m1[i]) != int(secret_message[i]):
                count += 1
        print('BER1:', count / len(m))

        for i in range(len(secret_message)):
            if int(m2[i]) != int(secret_message[i]):
                count += 1
        print('BER2:', count / len(m))
    # assert len(m) - len(stop_mark) == len(secret_message)
    # 判断：提取的信息长度是否等于 停止符号的长度+密文长度

def GetWaveLSB():
    wave_path = ChooseWAV()
    if wave_path == "":  # 啥也没选，直接退出吧！
        tkinter.messagebox.showinfo('提示', '请选择一个有效的源文件路径！')
        return
    stego_path = ChoosePath()
    if stego_path == "":  # 啥也没选，退出吧！
        tkinter.messagebox.showinfo('提示', '请选择一个有效的保存路径！')
        return

    bob = LSBExtractor(0)
    m = bob.extract(wave_path, stego_path+"\RecoverFromWav.txt")

def WAVLSB():
    root1 = Toplevel()
    root1.title("音频水印")
    root1.geometry('800x430')

    w = Canvas(root1)
    w.place(x=300, y=0, width=300, height=700)
    w.create_line(250, 50, 250, 370,
                  fill='#C0C0C0',
                  # fill='red',
                  width=2, )

    Label(root1, text="音频水印", font=fontStyle1).pack()
    button2 = Button(root1, text="LSB数字水印添加", command=DoWaveLSB)
    button0 = Button(root1, text="LSB数字水印提取", command=GetWaveLSB)

    Message(root1, text='∎通过LSB基本算法对WAV文件进行数字水印添加', cursor='cross',
            width='150').place(x=600, y=50)
    Message(root1, text='∎通过LSB基本算法对WAV文件进行数字水印提取', cursor='cross', width='150').place(
        x=600, y=170)

    button2.place(height=60, width=300, x=200, y=50)
    button0.place(height=60, width=300, x=200, y=170)
    root.mainloop()

def MP4LSB():
    root = Toplevel()

    Label(root, text="视频水印", font=fontStyle1).pack()
    root.title("视频水印")
    root.geometry('700x400')
    button3 = Button(root, text="LSB视频水印嵌入", command=GetFrameFromVideo)  # 控制label的边界
    button4 = Button(root, text="LSB视频水印提取", command=GetInfoFromVideo)  # 控制label的颜色
    button3.place(height=60, width=250, x=75, y=150)
    button4.place(height=60, width=250, x=350, y=150)

    Message(root, text='∎通过LSB基本算法对MP4文件进行数字水印添加', cursor='cross',
            width='150').place(x=100, y=250)
    Message(root, text='∎通过LSB基本算法对MP4文件进行数字水印提取', cursor='cross',
            width='150').place(x=430, y=250)
    root.mainloop()

np.random.seed(0)
root = Tk()  # 创建一个主窗体。相当于提供了一个搭积木的桌子
# center_window(root, 500, 200)
root.title("音频数字水印添加系统")
# root.geometry('1100x500+200+20')#调整窗体大小,第一个数横大小，第二个数纵大小，第三个数离左屏幕边界距离，第四个数离上面屏幕边界距离
root.geometry('850x500')  # 调整窗体大小,第一个数横大小，第二个数纵大小，第三个数离左屏幕边界距离，第四个数离上面屏幕边界距离

root.attributes('-toolwindow', False,
                '-alpha', 0.9,
                '-fullscreen', False,
                '-topmost', False)

global fontStyle
fontStyle = tkFont.Font(family="Lucida Grande", size=20)
fontStyle1 = tkFont.Font(family="Lucida Grande", size=15)
fontStyle2 = tkFont.Font(family="Lucida Grande", size=10)

w = Canvas(root)
w.place(x=500, y=170, width=300, height=190)

Label(root, text="音频数字水印保护系统", font=fontStyle).place(x=270, y=40)

style = Style(root)
style.configure("TButton", font=fontStyle)
style.configure("Test.TButton", font=fontStyle2)
Button(root, text='视频水印', command=MP4LSB).place(height=60, width=200, x=170, y=170)
Button(root, text='音频水印', command=WAVLSB).place(height=60, width=200, x=450, y=170)

Message(root, text='视频水印包含:\n    LSB水印嵌入和提取\n', cursor='cross', width='200').place(x=200, y=270, width=200)
Message(root, text='音频水印包含:\n    LSB隐写\n    LSB提取', cursor='cross', width='200').place(x=450, y=270, width=200)

root.mainloop()  # 开启一个消息循环队列，可以持续不断地接受操作系统发过来的键盘鼠标事件，并作出相应的响应
# mainloop应该放在所有代码的最后一行，执行他之后图形界面才会显示并响应系统的各种事件
'''
最终目标：实现基于LSB带UI的视频、音频（wav、mp3）数字水印添加和提取系统，用户可以自行选择电脑上的文件进行水印添加和提取
现在进展：基本实现了对wav文件的数字水印添加功能，实现了基于LSB算法的图片水印添加功能，实现了基本的UI
当前问题：数字水印的提取仍有待修改，冀望实现由输入的unicode自定义水印功能，mp3的数字水印添加有待实现
可以把图像水印添加时给定的水印信息长度进行填充后，进行加密，作为密钥返回给用户，提取时解密再进行水印提取
?
??
???
png√ jpg×
'''
