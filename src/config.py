import os

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_raw = os.path.join(current_dir, "data", "13560281")
path_filtered = os.path.join(current_dir, "filtered_data")
mat_files_raw = []
mat_files_filtered = []
n_harmonics = 3 #ile harmonicznych uwzględniamy w referencji
fs = 250 #w pracy napisali, że zdownsamplowali do 250Hz

target_freq = {
    0:9.25,
    1:11.25,
    2:13.25,
    3:9.75,
    4:11.75,
    5:13.75,
    6:10.25,
    7:12.25,
    8:14.25,
    9:10.75,
    10:12.75,
    11:14.75
}
