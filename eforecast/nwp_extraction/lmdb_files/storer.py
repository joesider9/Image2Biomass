import os
import shutil

def delete_tiff_files(output_path=None):

    try:
        for filename in os.listdir(f'{output_path}'):
            file_path = os.path.join(f'{output_path}', filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    except Exception as e:
        error_log_message = f'Could not Delete Grib Files due to {e}'
        print(error_log_message)
        shutil.rmtree(file_path)


