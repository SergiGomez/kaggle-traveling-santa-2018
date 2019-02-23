
# This script generates a folder with all the files needed for a TSP execution using
# LKH algorithm. The parameters have to be set in the prepare_files function.
# The next step is to run the execution itself. Should be something like the following:
# /home/luis/src/LKH-2.0.9/LKH temp/mat_.par

exec(open("./kaggle-competition-santa/utils.py").read())
import os

def write_tsp_file(np_xy_cities, folder, name='ts'):
    with open(folder + '/ts.tsp', 'w') as f:
        f.write('NAME : %s\n' % name)
        f.write('COMMENT : %s\n' % 'travelling santa')
        f.write('TYPE : TSP\n')
        f.write('DIMENSION : %d\n' % np_xy_cities.shape[0])
        f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
        f.write('NODE_COORD_SECTION\n')
        for i in range(np_xy_cities.shape[0]):
            x_, y_ = (np_xy_cities[i] * 1000).astype('int')
            text = str(i+1) + ' ' + str(x_) + ' ' + str(y_)
            f.write(text + '\n')


def write_parameters(parameters, folder):
    with open(folder + '/ts.par', 'w') as f:
        for param, value in parameters:
            f.write("{} = {}\n".format(param, value))
    print("Parameters saved as", folder + '/ts.par')


def prepare_files(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    print(folder + '/output.txt')
    cities = pd.read_csv('./input/cities.csv')
    np_xy_cities = cities[['X', 'Y']].values
    with open(folder + '/output.txt', 'w') as f:
        pass
    parameters = [
        ("PROBLEM_FILE", folder + "/ts.tsp"),
        ("OUTPUT_TOUR_FILE", folder + "/output.txt"),
        ("INITIAL_TOUR_FILE", folder + "/init.tour"),
        ("RUNS", 1),
        ("SEED", 139),
        ('CANDIDATE_SET_TYPE', 'POPMUSIC'), #'NEAREST-NEIGHBOR', 'ALPHA'),
        ("INITIAL_PERIOD", 10000),
        ("MAX_CANDIDATES", 7),
        ("MOVE_TYPE", 8),
        ("PATCHING_A", 4),
        ("PATCHING_C", 3)
    ]
    write_parameters(parameters, folder)
    write_tsp_file(np_xy_cities, folder=folder)
    print('~/LKH-2.0.9/LKH ' + folder + '/ts.par')

prepare_files(folder='1222_LKH_A4C3_init_local')


# os.system('/home/luis/src/LKH-2.0.9/LKH temp/mat_.par')
#
# path = np_read_output_LKH('temp/tsp_solution.csv')[:-1]
# path = subtour[path]

# exec(open("./kaggle-competition-santa/utils.py").read())
# continuous_score_calc(path = '1222_LKH_A4C3_init_local/', time_int = 600)
