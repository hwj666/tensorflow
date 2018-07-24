import xml.etree.ElementTree as ET

tree = ET.ElementTree(file='word.xml')
root = tree.getroot()

charset_file = open('charset.txt','w')
charset = set()
with open('anno.txt','w') as f:

    for image in root.findall('image'):
        path = image.attrib['file']
        label = image.attrib['tag']
        charset.update(list(label))
        f.write(path+' '+label+'\n')
charset_file.write(''.join(charset))