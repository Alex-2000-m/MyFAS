import os, random, shutil

def moveFile(fileDir, tarDir, picknumber, moname):
  """
  Parameter:
    fileDir: 原图片文件夹
    tarDir: 接受文件夹
    picknumber: 图片选取的数量
    moname: 要重命名的图片名字，在本函数中图片名字会被重命名为：{moname} + {count} + ".jpg"，
            其中".jpg"为图片的原格式。这个可以自己修改
  """
  pathDir = os.listdir(fileDir)
  sample = random.sample(pathDir, picknumber)
  count = 101
  for name in sample:
    shutil.move(fileDir + name, tarDir + name)
    os.rename(os.path.join(tarDir, name), os.path.join(tarDir,  str(count) + ".jpg"))
    count += 1


if __name__ == "__main__":
  fileDir = "D:/毕设/数据库论文/Oulu_Npu/test/1_1_38_2/"
  tarDir = "D:/毕设/数据库论文/神经网络/毕设用的神经网络/test/0/"
  moveFile(fileDir, tarDir, 50, 'val_image')
