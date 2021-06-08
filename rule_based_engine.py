from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow


def get_colors_ratio(img):
    width, height = img.size
    blue_count = 0
    yellow_count = 0
    red_count = 0
    green_count = 0
    low_green_count = 0
    orange_count = 0
    pixels = width * height
    for x in range(0, width):
        for y in range(0, height):
            if (255, 0, 0) == img.getpixel((x, y)) or (234, 21, 21) == img.getpixel((x, y)):  # RED
                red_count += 1
            elif (255, 128, 0) == img.getpixel((x, y)):  # ORANGE
                orange_count += 1
            elif (255, 255, 0) == img.getpixel((x, y)):  # YELLOW
                yellow_count += 1
            elif (0, 255, 255) == img.getpixel((x, y)) or (42, 212, 212) == img.getpixel((x, y)) or (46, 209, 209) == img.getpixel((x, y)) or (1, 254, 254) == img.getpixel((x, y)):  # BLUE
                blue_count += 1
            elif (0, 255, 0) == img.getpixel((x, y)):  # GREEN
                green_count += 1
            elif (128, 255, 0) == img.getpixel((x, y)):
                low_green_count += 1
            else:
                print(img.getpixel((x, y)))

    color_ratio = {'red': red_count / pixels * 100 * 5,
                   'orange': orange_count / pixels * 100 * 4,
                   'yellow': yellow_count / pixels * 100 * 3,
                   'blue': blue_count / pixels * 100 * 2,
                   'green': green_count / pixels * 100 * 2,
                   'low_green': low_green_count / pixels * 100 * 1}
    total = color_ratio.get('red') + color_ratio.get('orange') + color_ratio.get('yellow') + color_ratio.get(
        'blue') + color_ratio.get('green') + color_ratio.get('low_green')
    # print(total)
    return total


def crop(im, height, width):
    imgwidth, imgheight = im.size
    rows = np.int(imgheight / height)
    cols = np.int(imgwidth / width)
    for i in range(rows):
        for j in range(cols):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            yield im.crop(box)


def split_frames(image):
    #image.show()
    imgwidth, imgheight = image.size
    height = np.int(imgheight / 4)
    width = np.int(imgwidth / 3)
    start_num = 0
    frame_list = []
    for k, piece in enumerate(crop(image, height, width), start_num):
        if 5 < k < 9:
            img = Image.new('RGB', (width, height), 255)
            img.paste(piece)
            # filenum = 1
            # path = os.path.join("frame%d_1%05d.png" % (int(k+1), filenum))
            # img.save(path)
            # os.rename(path, os.path.join("frame%d_1%05d.png" % (int(k+1), filenum)))
            total_ratio = get_colors_ratio(img)
            if k == 7:
                frame_list.insert(0, (k, total_ratio))
            else:
                frame_list.append((k, total_ratio))
    return frame_list


"""
Steer Values(Left  -  Right)
Speed = 0 :  0.0         0.0
        1:  -0.7         0.7
        2:  -0.5         0.5
        3:  -0.3         0.3
        4:  -0.1         0.1
"""
steer_for_speed = {0: 0, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.1}
brake_for_speed = {0: 0, 1: 0.2, 2: 0.4, 3: 0.7, 4: 1.0}


def rule_based_engine(image, speed):
    print("Speed: ",speed)
    frames = split_frames(image)
    selected_side = sorted(frames, key=lambda frames: frames[1])[0]
    # TODO: steer değerleri belirlenecek
    # TODO: fren eklenecek (score değerine bakılarak) SCORE değeri olmadı. SPEEDe bak
    # TODO: gaz da çıkmalı
    if selected_side[0] == 6:  # GO LEFT
        steer = -steer_for_speed.get(speed)
    elif selected_side[0] == 7:  # GO STRAIGHT
        steer = 0.0
    elif selected_side[0] == 8:  # GO RIGHT
        steer = steer_for_speed.get(speed)
    if selected_side[1] > 350:
        brake = 1 * brake_for_speed.get(speed)
    elif selected_side[1] > 300:
        brake = 0.8 * brake_for_speed.get(speed)
    elif selected_side[1] > 250:
        brake = 0.7 * brake_for_speed.get(speed)
    elif selected_side[1] > 200:
        brake = 0.6 * brake_for_speed.get(speed)
    else:
        brake = 0.5 * brake_for_speed.get(speed)
    return steer, 0.0, brake
