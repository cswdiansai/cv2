import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import threading
import matplotlib.pyplot as plt

# 捕获摄像头
cap = cv2.VideoCapture(0)

# 设置摄像头参数，第一个和第二个为像素大小，第三个表示帧率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 17)

# 树莓派输出控制
EA, I2, I1, EB, I4, I3, LS, RS = (13, 19, 26, 16, 20, 21, 6, 5)
FREQUENCY = 50
GPIO.setmode(GPIO.BCM)
GPIO.setup([EA, I2, I1, EB, I4, I3], GPIO.OUT)
GPIO.setup([LS, RS], GPIO.IN)
GPIO.output([EA, I2, EB, I3], GPIO.LOW)
GPIO.output([I1, I4], GPIO.HIGH)

pwma = GPIO.PWM(EA, FREQUENCY)
pwmb = GPIO.PWM(EB, FREQUENCY)
pwma.start(0)
pwmb.start(0)

lspeed = 0
rspeed = 0
lcounter = 0
rcounter = 0


def delay_microseconds(us):
    start = pi.get_current_tick()
    while (pi.get_current_tick() - start) < us:
        pass


class PID:

    def __init__(self, P=80, I=0, D=0, speed=0, duty=26):

        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.err_pre = 0
        self.err_last = 0
        self.u = 0
        self.integral = 0
        self.ideal_speed = speed

    def update(self, feedback_value):
        self.err_pre = self.ideal_speed - feedback_value
        self.integral += self.err_pre
        self.u = self.Kp * self.err_pre + self.Ki * self.integral + self.Kd * (self.err_pre - self.err_last)
        self.err_last = self.err_pre
        if self.u > 100:
            self.u = 100
        elif self.u < 0:
            self.u = 0
        return self.u

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain


class PD:

    def __init__(self, P=0, D=0, location=240):  # 120

        self.Kp = P
        self.Kd = D
        self.err_pre = 0
        self.err_last = 0
        self.u = 0
        self.ideal_location = location

    def update(self, feedback_value):
        self.err_pre = feedback_value - self.ideal_location
        self.u = self.Kp * self.err_pre + self.Kd * (self.err_pre - self.err_last)
        self.err_last = self.err_pre
        if self.u > 100:
            self.u = 100
        return self.u

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain


def my_callback(channel):
    global lcounter
    global rcounter
    if (channel == LS):
        lcounter += 1
    elif (channel == RS):
        rcounter += 1


def getspeed():
    global rspeed
    global lspeed
    global lcounter
    global rcounter
    GPIO.add_event_detect(LS, GPIO.RISING, callback=my_callback)
    GPIO.add_event_detect(RS, GPIO.RISING, callback=my_callback)
    while True:
        rspeed = (rcounter / 585.0)
        lspeed = (lcounter / 585.0)
        rcounter = 0
        lcounter = 0
        time.sleep(0.1)


thread1 = threading.Thread(target=getspeed)
thread1.start()

l_origin_duty = 0
r_origin_duty = 0
pwma.start(l_origin_duty)
pwmb.start(r_origin_duty)
L_control = PID(40, 0, 10, 0, l_origin_duty)
R_control = PID(40, 0, 11, 0, r_origin_duty)

# 图像PD管理函数
Control = PD(0.002, 0.002, 160)

"""
在OpenCV中，HSV（色相、饱和度、明度）是一种常用的颜色空间，它可以方便地进行颜色的识别和处理。
以下是红、绿和蓝颜色在HSV空间中的范围：

红色（Red）：
色相（Hue）范围：0-10 和 170-180
饱和度（Saturation）范围：50-255
明度（Value）范围：50-255
这是我测试代码时用的颜色范围：
lower_color = np.array([0, 150, 150])/higher_color = np.array([10, 255, 255])
q
黄色（Yellow）
色相（Hue）范围：20-40
饱和度（Saturation）范围：50-255
明度（Value）范围：50-255

这是我测试代码时用的颜色范围：
lower_color = np.array([20, 70, 50])/higher_color = np.array([40, 200, 150])

绿色（Green）：
色相（Hue）范围：60-80
饱和度（Saturation）范围：50-255
明度（Value）范围：50-255
lower_color = np.array([60, 70, 50])
higher_color = np.array([80, 200, 255])

蓝色（Blue）：
色相（Hue）范围：90-130
饱和度（Saturation）范围：50-255
明度（Value）范围：50-255

"""

Red_Eable = 3
Yellow_Eable = 4
Green_Eable = 5

# 预定义公共内核
kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def color_processing(hsv_img, lower, upper, max_contours=1):
    """通用颜色处理函数"""
    mask = cv2.inRange(hsv_img, lower, upper)
    dilate = cv2.dilate(mask, kernel_5x5, iterations=1)

    _, cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not cnts:
        return ()  # 返回空元组

    # 优化轮廓查找逻辑
    if max_contours == 1:
        c = max(cnts, key=cv2.contourArea)
        return cv2.boundingRect(c)

    # 查找前N大轮廓的优化实现
    areas = [(cv2.contourArea(c), c) for c in cnts]
    areas.sort(reverse=True, key=lambda x: x[0])
    return [cv2.boundingRect(c) for _, c in areas[:max_contours]]


# 修改后的颜色识别函数（保持原有注释不变）
def Red_Identify(hsv):
    if Red_Eable != 3:
        return 0, 0, 0, 0

    # 二值化处理，表示HSV中颜色的范围
    lower_color = np.array([0, 50, 50])
    higher_color = np.array([10, 255, 255])

    result = color_processing(hsv, lower_color, higher_color)
    return result if result else (0, 0, 0, 0)


def Yellow_Identify(hsv):
    if Yellow_Eable != 3:
        return 0, 0, 0, 0

    # 二值化处理，表示HSV中颜色的范围
    lower_color = np.array([20, 65, 120])
    higher_color = np.array([40, 230, 230])

    result = color_processing(hsv, lower_color, higher_color)
    return result if result else (0, 0, 0, 0)


def Greem_Identify(hsv):
    if Green_Eable != 3:
        return 0, 0, 0, 0, 0, 0, 0, 0

    # 二值化处理，表示HSV中颜色的范围
    lower_color = np.array([60, 45, 50])
    higher_color = np.array([80, 205, 235])

    results = color_processing(hsv, lower_color, higher_color, max_contours=2)
    if not results:
        return (0, 0, 0, 0, 0, 0, 0, 0)

    if len(results) == 1:
        x, y, w, h = results[0]
        return (x, y, w, h, 0, 0, 0, 0)

    # 按面积排序返回结果
    (x1, y1, w1, h1), (x2, y2, w2, h2) = sorted(results,
                                                key=lambda r: r[2] * r[3],
                                                reverse=True)
    return (x1, y1, w1, h1, x2, y2, w2, h2)


def MaxClolor_Identify(img, hsv):
    # 优先检测红色
    x0, y0, w0, h0 = Red_Identify(hsv)
    if w0 * h0 > 500:  # 红色检测成功
        # 绘制红色框
        cv2.rectangle(img, (x0, y0), (x0 + w0, y0 + h0), (0, 0, 255), 3)
        cv2.circle(img, (x0 + (w0 // 2), y0 + (h0 // 2)), 6, (0, 0, 255), 2)
        cv2.imshow("identify", img)
        return 1, x0, w0, h0

    # 红色未检测到，检测黄色
    x1, y1, w1, h1 = Yellow_Identify(hsv)
    if w1 * h1 > 800:  # 黄色检测成功
        # 绘制黄色框
        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 255), 3)
        cv2.circle(img, (x1 + (w1 // 2), y1 + (h1 // 2)), 6, (0, 255, 255), 2)
        cv2.imshow("identify", img)
        return 2, x1, w1, h1

    # 黄红均未检测到，检测绿色
    gx1, gy1, gw1, gh1, gx2, gy2, gw2, gh2 = Greem_Identify(hsv)
    if gw1 * gh1 > 800:  # 绿色检测成功
        # 绘制绿色框（可能两个）
        cv2.rectangle(img, (gx1, gy1), (gx1 + gw1, gy1 + gh1), (0, 255, 0), 3)
        cv2.circle(img, (gx1 + (gw1 // 2), gy1 + (gh1 // 2)), 6, (0, 255, 0), 2)
        if gw2 * gh2 > 800:  # 存在第二个绿色区域
            cv2.rectangle(img, (gx2, gy2), (gx2 + gw2, gy2 + gh2), (0, 255, 0), 3)
            cv2.circle(img, (gx2 + (gw2 // 2), gy2 + (gh2 // 2)), 6, (0, 255, 0), 2)
            cv2.imshow("identify", img)
            return 3, gx1, 0, gh1
        cv2.imshow("identify", img)
        return 3, gx1, gx2, gh1

    # 无颜色检测到
    cv2.imshow("identify", img)
    return 0, 0, 0, 0


Turn_flag = 0


def PID_Set(img, hsv):
    MAXH = 160
    global Turn_flag

    global Red_Eable
    global Yellow_Eable
    global Green_Eable

    flag, x1, w1, h = MaxClolor_Identify(img, hsv)
    if (Red_Eable * Yellow_Eable * Green_Eable == 0):

        pwma.ChangeDutyCycle(40)
        pwmb.ChangeDutyCycle(40)

    elif (flag == 1):
        Control.ideal_location = 160
        if (h > MAXH):

            pwma.ChangeDutyCycle(0)
            pwmb.ChangeDutyCycle(40)

            time.sleep(0.4)

            pwma.ChangeDutyCycle(45)
            pwmb.ChangeDutyCycle(45)
            time.sleep(0.4)

            Turn_flag = 1
            Red_Eable = Red_Eable - 1
            Yellow_Eable = Yellow_Eable - 1
            Green_Eable = Green_Eable - 1

            pwma.ChangeDutyCycle(40)
            pwmb.ChangeDutyCycle(22)


        else:

            L_control.ideal_speed = 0.9 + Control.update(x1 + w1 / 2)
            R_control.ideal_speed = 0.9 - Control.update(x1 + w1 / 2)

            pwma.ChangeDutyCycle(L_control.update(lspeed))
            pwmb.ChangeDutyCycle(R_control.update(rspeed))
    elif (flag == 2):
        Control.ideal_location = 160

        if (h > MAXH - 10):

            pwma.ChangeDutyCycle(40)
            pwmb.ChangeDutyCycle(0)

            time.sleep(0.5)

            pwma.ChangeDutyCycle(45)
            pwmb.ChangeDutyCycle(45)
            time.sleep(0.5)

            Turn_flag = 2
            Red_Eable = Red_Eable - 1
            Yellow_Eable = Yellow_Eable - 1
            Green_Eable = Green_Eable - 1

            pwma.ChangeDutyCycle(20)
            pwmb.ChangeDutyCycle(40)


        else:

            L_control.ideal_speed = 0.9 + Control.update(x1 + w1 / 2)
            R_control.ideal_speed = 0.9 - Control.update(x1 + w1 / 2)

            pwma.ChangeDutyCycle(L_control.update(lspeed))
            pwmb.ChangeDutyCycle(R_control.update(rspeed))
    elif (flag == 3):
        if (Turn_flag == 1):
            if (w1 != 0):
                Control.ideal_location = 360

                L_control.ideal_speed = 0.9 + Control.update(x1 / 2 + w1 / 2)
                R_control.ideal_speed = 0.9 - Control.update(x1 / 2 + w1 / 2)

                pwma.ChangeDutyCycle(L_control.update(lspeed))
                pwmb.ChangeDutyCycle(R_control.update(rspeed))
            else:
                if (h > MAXH - 15):
                    pwma.ChangeDutyCycle(35)
                    pwmb.ChangeDutyCycle(0)

                    time.sleep(0.5)

                    pwma.ChangeDutyCycle(40)
                    pwmb.ChangeDutyCycle(40)
                    time.sleep(0.4)

                    Red_Eable = Red_Eable - 1
                    Yellow_Eable = Yellow_Eable - 1
                    Green_Eable = Green_Eable - 1

                    pwma.ChangeDutyCycle(20)
                    pwmb.ChangeDutyCycle(40)
                    time.sleep(0.3)

                else:
                    Control.ideal_location = 140

                    L_control.ideal_speed = 0.9 + Control.update(x1)
                    R_control.ideal_speed = 0.9 - Control.update(x1)

                    pwma.ChangeDutyCycle(L_control.update(lspeed))
                    pwmb.ChangeDutyCycle(R_control.update(rspeed))
        if (Turn_flag == 2):
            if (w1 != 0):
                Control.ideal_location = -40

                L_control.ideal_speed = 0.9 + Control.update(x1 / 2 + w1 / 2)
                R_control.ideal_speed = 0.9 - Control.update(x1 / 2 + w1 / 2)

                pwma.ChangeDutyCycle(L_control.update(lspeed))
                pwmb.ChangeDutyCycle(R_control.update(rspeed))
            else:
                if (h > MAXH - 15):
                    pwma.ChangeDutyCycle(0)
                    pwmb.ChangeDutyCycle(35)

                    time.sleep(0.5)

                    pwma.ChangeDutyCycle(45)
                    pwmb.ChangeDutyCycle(45)
                    time.sleep(0.4)

                    Red_Eable = Red_Eable - 1
                    Yellow_Eable = Yellow_Eable - 1
                    Green_Eable = Green_Eable - 1
                    Turn_flag = 1

                    pwma.ChangeDutyCycle(40)
                    pwmb.ChangeDutyCycle(22)
                    time.sleep(0.3)

                else:
                    Control.ideal_location = 180

                    L_control.ideal_speed = 0.9 + Control.update(x1)
                    R_control.ideal_speed = 0.9 - Control.update(x1)

                    pwma.ChangeDutyCycle(L_control.update(lspeed))
                    pwmb.ChangeDutyCycle(R_control.update(rspeed))

    else:
        a = 0


while True:
    # 循环读取每一帧
    res, img = cap.read()

    # 转换为HSV颜色模型
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 获取图像并调整PID
    PID_Set(img, hsv)

    key = cv2.waitKey(1) & 0xFF  # 检测键盘,最长等待1ms
    if key == ord('q'):
        break  # 按q时结束

pwma.stop()
pwmb.stop()
GPIO.cleanup()

# 关闭摄像头，解除程序占用摄像头
cap.release()

# cv2把所有打开的窗口关闭掉
cv2.destroyAllWindows()
