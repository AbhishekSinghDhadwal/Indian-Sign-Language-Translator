from PIL import Image, ImageEnhance, ImageFilter
import random
import os

set_count = 3 #number of bright/dark images per picture
set_blur_count = 2 #number of blurred pics per bright/dark image

brightness_range = 2 #upper limit for image brightening
blur_lower_radius = 5
blur_upper_radius = 20

for root, dirs, files, in os.walk('dataset'):
    if files != []:
        parent_path = str(root)
        pos = parent_path.rfind("\\")
        if pos == -1:
            pos = parent_path.rfind("/")
        parent_dir = parent_path[pos + 1:]
        for i in files:
            input_filename = str(i)
            extension = input_filename[-4:]
            input_path = os.path.join(parent_path, input_filename)
            img = Image.open(input_path)
            enhancer = ImageEnhance.Brightness(img)
            img_count = 1 #constantly incremented, appends _1, _2 etc to the end of the file name
            
            #darken the image
            for i in range(1, set_count + 1):
                factor = random.uniform(0.1, 0.9) #use random number as a darkening factor
                img_output = enhancer.enhance(factor)
                output_path = 'modded_' + input_path
                output_path = output_path[:-4]
                output_path = output_path + '_' + str(img_count) + extension #set file name with appropriate naming convention
                img_count += 1
                img_output.save(output_path)
            
                #blur each darkened image
                for x in range(1, set_blur_count + 1):
                    get_img = Image.open(output_path)
                    blur_val = random.randint(blur_lower_radius, blur_upper_radius) #use random number for blurring
                    blur_img = get_img.filter(ImageFilter.BoxBlur(blur_val))
                    pos1 = output_path.rfind("_")
                    blur_output_path = output_path[:pos1]
                    blur_output_path = blur_output_path + '_' + str(img_count) + extension
                    img_count += 1
                    blur_img.save(blur_output_path)
            
            #brighten the image
            for i in range(1, set_count + 1):
                factor = random.uniform(1.1, brightness_range)
                img_output = enhancer.enhance(factor)
                output_path = 'modded_' + input_path
                output_path = output_path[:-4]
                output_path = output_path + '_' + str(img_count) + extension #set file name with appropriate naming convention
                img_count += 1
                img_output.save(output_path)
            
                #blur each brightened image
                for x in range(1, set_blur_count + 1):
                    get_img = Image.open(output_path)
                    blur_val = random.randint(blur_lower_radius, blur_upper_radius) #use random number for blurring
                    blur_img = get_img.filter(ImageFilter.BoxBlur(blur_val))
                    pos1 = output_path.rfind("_")
                    blur_output_path = output_path[:pos1]
                    blur_output_path = blur_output_path + '_' + str(img_count) + extension
                    img_count += 1
                    blur_img.save(blur_output_path)