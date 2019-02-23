# from optimization import *
exec(open("./kaggle-competition-santa/optimization.py").read())

cities = pd.read_csv('./input/cities.csv')
np_xy_cities = cities[['X', 'Y']].values.astype('float')


directory = '1229_kotya_copy'


numba_2opt(filename=directory+'/output.txt', path_out=directory+'/sub_2opt')


file_temp = [file for file in os.listdir(directory) if '2opt' in file][0]
filename = directory + '/' + file_temp
path_out = directory + '/' + '3comb'
numba_3comb(filename=filename, path_out=path_out)

file_temp = [file for file in os.listdir(directory) if '3comb' in file][0]
filename = directory + '/' + file_temp
path_out = directory + '/' + 'sub_'
subtram_lkh(file_path=filename, path_out=path_out, tempdir='temp5)

file_temp = [file for file in os.listdir(directory) if '500' in file][0]
filename = directory + '/' + file_temp
path_out = directory + '/' + '4comb'
numba_4comb(filename=filename, path_out=path_out)

file_temp = [file for file in os.listdir(directory) if '4comb' in file][0]
filename = directory + '/' + file_temp
path_out = directory + '/' + '5comb1'
numba_5comb(filename=filename, path_out=path_out)

file_temp = [file for file in os.listdir(directory) if '5comb1' in file][0]
filename = directory + '/' + file_temp
path_out = directory + '/' + '5comb2'
numba_5comb2(filename=filename, path_out=path_out)

file_temp = [file for file in os.listdir(directory) if '5comb2' in file][0]
filename = directory + '/' + file_temp
path_out = directory + '/' + 'sub2_'
subtram_lkh(file_path=filename, path_out=path_out, tempdir='temp6')

file_temp = [file for file in os.listdir(directory) if 'sub2_500' in file][0]
filename = directory + '/' + file_temp
n=14
path_out = directory + '/' + 'sub_'+str(n)+'shuff_score.csv'
cmd = 'time ./kaggle-competition-santa/dp '+str(n)+' <./'+filename+' >'+path_out
os.system(cmd)

file_temp = [file for file in os.listdir(directory) if '14shuff_' in file][0]
filename = directory + '/' + file_temp
path_out = directory + '/' + '7comb'
numba_7comb(filename=filename, path_out=path_out)

file_temp = [file for file in os.listdir(directory) if '7comb' in file][0]
filename = directory + '/' + file_temp
n=15
path_out = directory + '/' + 'sub_'+str(n)+'shuff_score.csv'
cmd = 'time ./kaggle-competition-santa/dp '+str(n)+' <./'+filename+' >'+path_out
os.system(cmd)

file_temp = [file for file in os.listdir(directory) if '15shuff_' in file][0]
filename = directory + '/' + file_temp
path_out = directory + '/' + '2lkhrec'
subtram_lkh_rec(file_path=filename, path_out=path_out, tempdir='temp3', multiply=1.2)

file_temp = [file for file in os.listdir(directory) if '2lkhrec' in file][0]
filename = directory + '/' + file_temp
path_out = directory + '/' + '3lcomb'
numba_3comb_large(filename=filename, path_out=path_out)

file_temp = [file for file in os.listdir(directory) if '3lcomb' in file][0]
filename = directory + '/' + file_temp
path_out = directory + '/' + 'lkhrec2new'
subtram_lkh_rec2(file_path=filename, path_out=path_out, max_i=1000, tempdir='temp3',
                    sub_size = 200, loops = 5)
# subtram_lkh_rec2(file_path=filename, path_out=path_out, max_i=2000, tempdir='temp2',
#                     sub_size = 100, loops = 5)

file_temp = [file for file in os.listdir(directory) if 'lkhrec2new' in file][0]
filename = directory + '/' + file_temp
n=16
path_out = directory + '/' + 'sub_'+str(n)+'shuff_score.csv'
cmd = 'time ./kaggle-competition-santa/dp '+str(n)+' <./'+filename+' >'+path_out
os.system(cmd)
