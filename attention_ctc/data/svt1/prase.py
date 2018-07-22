from PIL import Image
import os

import xml.etree.ElementTree as ET

save_path = 'images/'

tree = ET.ElementTree(file='train.xml')
root = tree.getroot()
i = 0

min_width = 20

charset_file = open('charset.txt','w')
charset = set()

with open('anno.txt','w') as f:
    for image in root.findall('image'):
        imageName = image.find('imageName')
        img = Image.open(imageName.text).convert('L')

        taggedRectangles = image.find('taggedRectangles')
        for taggedRectangle in taggedRectangles:
            location = taggedRectangle.attrib
            height = int(location['height'])
            width = int(location['width'])
            x, y = int(location['x']), int(location['y'])
            anno = taggedRectangle.find('tag')
            anno = anno.text

            new_img = img.crop((x,y,x+width,y+height))
            widht,height = new_img.size
            new_height = 32
            if height < 2 or width < 2:
                continue
            new_width = max(new_height * width // height,min_width)
            new_img = new_img.resize((new_width, new_height),Image.ANTIALIAS)
            new_name = save_path + str(i) + '.jpg'
            f.write(new_name+ ' ' + anno + '\n')
            new_img.save(new_name)
            i+=1
            charset.update(list(anno))
charset_file.write(''.join(charset))
charset_file.close()