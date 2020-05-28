import  imageio
import os,sys
def get_files(dir="./show_time/"):
    png_files =[]
    for root, dirs, files in os.walk(dir):
        for each in files:
            if '.png' in each:
                png_files.append(dir+each)
    #png_files = sorted(png_files)
    return png_files

if __name__ == '__main__':
    png_files = get_files("./show_time_gpu/")
    print(png_files)
    gif_images = []
    for each in png_files:
        gif_images.append(imageio.imread(each))
    imageio.mimsave('showtime.gif',gif_images,fps=10)