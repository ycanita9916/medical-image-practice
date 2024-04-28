import argparse
import numpy as np
import SimpleITK as sitk
import os
import sys
import gzip
import shutil

def createMIP(np_img, sitk_img, thickness_mm, overlap_mm,projection_type):
    # Get image info
    z_length_px = np_img.shape[0]
    pixel_spacing_mm = sitk_img.GetSpacing()[2]
    
    slice_thickness = round(thickness_mm / pixel_spacing_mm)
    overlap = round(overlap_mm / pixel_spacing_mm)
    slice_step = slice_thickness - overlap
    slices_num = round(z_length_px / slice_step)
    
    # Calculate new spacing for MIP
    new_spacing_z = pixel_spacing_mm * z_length_px / slices_num
    new_spacing_list = list(sitk_img.GetSpacing())
    new_spacing_list[2] = new_spacing_z
    
    # Empty MIP array
    img_shape = (slices_num,) + np_img.shape[1:]
    np_mip = np.zeros(img_shape)
    
    # Generate MIP
    for i in range(img_shape[0]):
        start = max(0, i * slice_step)
        end = min(z_length_px, start + slice_thickness)
        # np_mip[i, :, :] = np.amax(np_img[start:end], axis=0)
        if projection_type == 'MaxIP':
            np_mip[i, :, :] = np.amax(np_img[start:end], axis=0)
        elif projection_type == 'MinIP':
            np_mip[i, :, :] = np.amin(np_img[start:end], axis=0)
        
    return np_mip, new_spacing_list

def main():
    parser = argparse.ArgumentParser(description="Create Maximum Intensity Projection (MIP) from a 3D image.")
    parser.add_argument("--input", type=str,required=True, help="Input NIfTI file path")
    parser.add_argument("--output", type=str, required=True, help="Output NIfTI file path")
    parser.add_argument("--thickness_mm", type=float, default=10, help="Thickness of each slice in mm")
    parser.add_argument("--overlap_mm", type=float, default=5, help="Overlap between adjacent slices in mm")
    parser.add_argument("--projection_type", choices=['MaxIP', 'MinIP'], default='MaxIP', help="Type of projection to create (MaxIP or MinIP)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit("Input file not found: {}".format(args.input))

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        sys.exit("Output directory does not exist: {}".format(output_dir))

    else:
        input_path = args.input

    # Read IMG
    sitk_img = sitk.ReadImage(args.input)
    np_img = sitk.GetArrayFromImage(sitk_img)

    np_mip, new_spacing_list = createMIP(np_img, sitk_img, args.thickness_mm, args.overlap_mm, args.projection_type)

    # From MIP np array to IMG
    sitk_mip = sitk.GetImageFromArray(np_mip)

    sitk_mip.SetSpacing(new_spacing_list)
    sitk_mip.SetDirection(sitk_img.GetDirection())
    sitk_mip.SetOrigin(sitk_img.GetOrigin())

    writer = sitk.ImageFileWriter()
    writer.SetFileName(args.output + '.gz')
    writer.Execute(sitk_mip)

if __name__ == "__main__":
    main()
