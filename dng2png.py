import rawpy
import imageio
    
def convert(img_path):
    output_dir = './original/'
    
    with rawpy.imread(img_path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=False,
                                  no_auto_bright=True, output_bps=16)
    imageio.imsave(output_dir+'input.png', rgb)
    return output_dir + 'input.png'

if __name__== '__main__':
    convert('/home/saahil/rps/Hackvento2k19/nictovision/uploaded_img.dng')