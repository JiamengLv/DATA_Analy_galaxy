import pandas as pd
import urllib.request
import os
from concurrent.futures import ThreadPoolExecutor

def download_url(url, filename):
    if not os.path.exists(filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")


def generate_filenames(row, savepath):
    """Generate filenames for each type of data."""
    filename = row['TARPNUM']
    Timelabel = row['T_REC'].replace(':', '.')

    magnetname = f"{savepath}/magnetogram/{filename}/{Timelabel}_magnetogram.fits"
    continuumname = f"{savepath}/continuum/{filename}/{Timelabel}_continuum.fits"
    brname = f"{savepath}/br/{filename}/{Timelabel}_br.fits"
    bpname = f"{savepath}/bp/{filename}/{Timelabel}_bp.fits"
    btname = f"{savepath}/bt/{filename}/{Timelabel}_bt.fits"

    return {
        'magnetogram': magnetname,
        'continuum': continuumname,
        'br': brname,
        'bp': bpname,
        'bt': btname
    }


def parallel_download(data_info, savepath):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for index, row in data_info.iterrows():
            urls = {
                'magnetogram': row['magnetogram'],
                'continuum': row['continuum'],
                'br': row['Br'],
                'bp': row['Bp'],
                'bt': row['Bt']
            }
            filenames = generate_filenames(row, savepath)

            for key in urls:
                url = urls[key]
                filename = filenames[key]
                futures.append(executor.submit(download_url, url, filename))

        for future in futures:
            future.result()

if __name__ == '__main__':

    savepath = './HMI'
    hmi_csv = './filtered_hmi_data_url.csv'

    data_info = pd.read_csv(hmi_csv)
    parallel_download(data_info, savepath)


# import pandas as pd
# import os

# def generate_filenames(row, savepath):
#     """Generate filenames for each type of data."""
#     filename = row['TARPNUM']
#     Timelabel = row['T_REC'].replace(':', '.')
    
#     magnetname = f"{savepath}/magnetogram/{filename}/{Timelabel}_magnetogram.fits"
#     continuumname = f"{savepath}/continuum/{filename}/{Timelabel}_continuum.fits"
#     brname = f"{savepath}/br/{filename}/{Timelabel}_br.fits"
#     bpname = f"{savepath}/bp/{filename}/{Timelabel}_bp.fits"
#     btname = f"{savepath}/bt/{filename}/{Timelabel}_bt.fits"
    
#     return {
#         'magnetogram': magnetname,
#         'continuum': continuumname,
#         'br': brname,
#         'bp': bpname,
#         'bt': btname
#     }

# if __name__ == '__main__':
#     savepath = './HMI'
#     hmi_csv = './hmi_data_url.csv'
#     output_csv = './filtered_hmi_data_url.csv'  # 新的csv文件路径
    
#     data_info = pd.read_csv(hmi_csv)
    
#     rows_to_keep = []
#     for index, row in data_info.iterrows():
#         filenames = generate_filenames(row, savepath)
#         if not all([os.path.exists(f) for f in filenames.values()]):
#             # 如果所有fits文件都不存在，则保留这行数据
#             rows_to_keep.append(row)
    
#     filtered_data_info = pd.DataFrame(rows_to_keep)
#     filtered_data_info.to_csv(output_csv, index=False)