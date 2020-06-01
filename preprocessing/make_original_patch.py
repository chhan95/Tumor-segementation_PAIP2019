import openslide
import os
import re
import PIL

dataset_path='/media/disk/han/dataset/'
svs_tif_path=dataset_path+'svs_tif/'
original_patch_path=dataset_path+'original_patch'
cancer_viable_label_path=dataset_path+'cancer_viable_label_patch'
for r,_,filenames in os.walk(svs_tif_path):
    for filename in filenames:
        if re.search("\.(svs)$",filename) or re.search("\.(SVS)$",filename):
            image_path=r+'/'+filename
            img=openslide.OpenSlide(image_path)
            w,h=img.level_dimensions[0]


            if not os.path.exists(TEST_PAIP_path+part_path[2]):
                os.mkdir(TEST_PAIP_path+part_path[2])

            for i in range(0,h,1024):
                for j in range(0,w,1024):
                    patch=img.read_region((j,i),0,(1024,1024)).convert('RGB')
                    path=TEST_PAIP_path+part_path[2]+'/'+str(i//1024)+'_'+str(j//1024)+'.jpg'
                    patch.save(path)
