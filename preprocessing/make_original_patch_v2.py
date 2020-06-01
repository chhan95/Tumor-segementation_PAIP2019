import openslide
import os
import re
import PIL

PAIP_original_patch_path = 'D:/validation_set/'
TEST_PAIP_path = 'D:/TEST_PAIP/'
for r, _, filenames in os.walk(PAIP_original_patch_path):
    for filename in filenames:
        if re.search("\.(svs)$", filename) or re.search("\.(SVS)$", filename):
            image_path = r + '/' + filename
            img = openslide.OpenSlide(image_path)
            w, h = img.level_dimensions[1]

            part_path = r.split('/')
            if not os.path.exists(TEST_PAIP_path + part_path[2]):
                os.mkdir(TEST_PAIP_path + part_path[2])

            patch = img.read_region((0,0), 1, (w, h)).convert('RGB')
            path = TEST_PAIP_path + part_path[2] + '/' + filename[:-4] + '.jpg'
            patch.save(path)
