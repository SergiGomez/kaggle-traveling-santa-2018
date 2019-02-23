
# from utils import *
# exec(open("./kaggle-competition-santa/optimization.py").read())
exec(open("./kaggle-competition-santa/utils.py").read())

from matplotlib import collections  as mc
import numpy as np
import pandas as pd
from itertools import permutations
from sympy import isprime, primerange
import time

# functions definitions

def get_group_score(actual_path, groups, swap, offset = 0):

    # original data 1D
    np_xy_cities = cities[['X', 'Y']].values
    list_path = actual_path[offset:(len(actual_path)-(len(actual_path)-offset)%groups)] #np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])#df_submit['Path'].values

    # create 1D variables
    len_path = len(list_path)
    max_path = np.max(list_path)
    np_is_non_prime = np.ones(max_path + 1)
    list_prime = [i for i in primerange(0, len_path+1)]
    np_is_non_prime[list_prime] = 0
    np_is_path_from_non_prime = np.concatenate([np_is_non_prime[list_path][:-1], np.array([0])])
    position = np.arange(offset, offset+len_path)

    # transform to 2D
    list_path = list_path.reshape(-1, groups)
    position = position.reshape(-1, groups)
    np_is_path_from_non_prime = np_is_path_from_non_prime.reshape(-1, groups)

    list_path = list_path[:, swap]  # np.stack([list_path[:, 0], list_path[:, 2], list_path[:, 1], list_path[:, 3]], axis=1)
    #position = position[:, swap]  # np.stack([position[:, 0], position[:, 2], position[:, 1], position[:, 3]], axis=1)
    np_is_path_from_non_prime = np_is_path_from_non_prime[:, swap]  # np.stack([np_is_path_from_non_prime[:, 0], np_is_path_from_non_prime[:, 2], np_is_path_from_non_prime[:, 1], np_is_path_from_non_prime[:, 3]], axis=1)

    np_is_path_from_non_prime = np_is_path_from_non_prime[:, :-1]

    # Calc Distance
    np_xy_path = np_xy_cities[list_path]
    np_dist_path = np.sum((np_xy_path[:,:-1] - np_xy_path[:, 1:]) ** 2, axis=2) ** 0.5


    # Flag np.array, is path number(1 start) % 10 == 0?
    np_is_path_num_per_ten = ((position[:,1:])%10==0)*1 #position//10 #np.array(([0] * 9 + [1]) * ((len_path - 1) // 10) + [0] * ((len_path - 1) % 10))

    np_dist = np_dist_path * (1.0 + 0.1 * np_is_path_from_non_prime * np_is_path_num_per_ten)
    return np.sum(np_dist, axis = 1), list_path

def check_max_dist(num_list):
    num_list = np.array(num_list)
    length = len(num_list)
    max_dist = np.abs(np.arange(length)[num_list-1] - np.arange(length)).max()
    return max_dist

# 1: Optimization divinding current path into many subpaths og length less than 10 and trying all
# the order combinations to find the optimal one

def get_all_combinations(actual_path, groups, offset = 0, min_dist_diff = 0):
    gr_it = groups -2
    iterable = np.arange(gr_it)+1

    perm_tmp = list(permutations(iterable, gr_it))
    perm = []
    for perm_ in perm_tmp:
        max_dist = check_max_dist(perm_)
        if max_dist >= min_dist_diff:
            perm_ = list(perm_)
            perm_.insert(0, 0)
            perm_.append(groups-1)
            perm.append(perm_)
    print('number of iterations: '+str(len(perm)))

    best_dist = np.array([])

    for perm_ in perm:
        dist, path = get_group_score(actual_path, groups = groups, swap = perm_, offset = offset)
        if best_dist.shape[0] == 0:
            best_dist = dist
            best_path = path
            actual_dist = dist.sum()
        else:
            index = np.argmin([best_dist, dist], axis=0)
            best_dist = np.stack([best_dist, dist])[index, np.arange(len(dist))]
            best_path = np.stack([best_path, path])[index, np.arange(best_path.shape[0])]

    #possible_dist = np.vstack(possible_dist)
    #possible_paths = np.stack(possible_paths, axis = 0)

    best_tot_dist = best_dist.sum()

    distance_saved = actual_dist - best_tot_dist
    print(distance_saved)

    optimized_path_temp = best_path.reshape(-1)

    optimized_path = actual_path

    optimized_path[offset:(offset + len(optimized_path_temp))] = optimized_path_temp

    print(calc_score(optimized_path, cities))

    return optimized_path

# # example of use (uncomment)
# actual_path = np_read_output_LKH('1129_2_LKH_local/output.txt')
# cities = pd.read_csv('./input/cities.csv')
#
# print(calc_score(actual_path, cities))
# optimized_path = actual_path
#
# x = time.time()
# variables = [[9, 0], [9, 3]]
# for variable in variables:
#     groups, offset = variable
#     optimized_path = get_all_combinations(optimized_path, groups=groups, offset=offset)
# print(str(time.time() -x)+' seconds')
# print(str((time.time() -x)/60)+' minutes')
#
# calc_score(optimized_path, cities)
#
# pd.DataFrame({'Path': optimized_path}).to_csv('1129_2_LKH_local/2_opt_16413.csv', index=False)



# 2: deprecated, use number 3


def opt_path_2opt(sub, cities):
    # 2-opt check possible changes and benefit
    print('current score is: {0:.0f}'.format(calc_score(sub, cities)))

    city_coords = cities[['X', 'Y']].values
    path_ids_rep = np.repeat(sub[:-1], 5)
    path_mid_coord = calc_mid_coord(sub, cities)

    tree = KDTree(path_mid_coord)
    neighbor_dists, neighbor_indices = tree.query(path_mid_coord, k=6)
    neighbors_flat = neighbor_indices[:, 1:].reshape(-1)
    neighbors_flat = sub[neighbors_flat]
    print("Constructed tree with {} neighbors".format(neighbors_flat.shape[0]))

    scores = list()
    while True:
        path_df = pd.DataFrame({'Path': (sub[:-1])})
        path_df.set_index('Path', inplace=True)

        path_df['path_after'] = sub[1:]
        path_df.loc[:, 'X'] = city_coords[path_df.index][:, 0]
        path_df.loc[:, 'Y'] = city_coords[path_df.index][:, 1]
        path_df.loc[:, 'X_next'] = city_coords[path_df.path_after][:, 0]
        path_df.loc[:, 'Y_next'] = city_coords[path_df.path_after][:, 1]

        # path_df['dist'] = calc_dists(sub, cities)
        path_df['position'] = np.arange(path_df.shape[0])
        path_complete = path_df.loc[path_ids_rep]
        path_complete.loc[:, 'neighbor'] = neighbors_flat
        path_complete.loc[:, 'nei_position'] = path_df.loc[path_complete.neighbor].position.values
        path_complete.loc[:, 'neighbor_after'] = path_df.loc[path_complete['neighbor'], 'path_after'].values

        path_complete.loc[:, 'X_nei'] = path_df.loc[path_complete.neighbor].X.values
        path_complete.loc[:, 'Y_nei'] = path_df.loc[path_complete.neighbor].Y.values
        path_complete.loc[:, 'X_nei_next'] = path_df.loc[path_complete.neighbor_after].X.values
        path_complete.loc[:, 'Y_nei_next'] = path_df.loc[path_complete.neighbor_after].Y.values

        # filter
        path_complete = path_complete[path_complete.path_after != path_complete.neighbor]
        path_complete = path_complete[path_complete.index.values != path_complete.neighbor_after.values]

        path_complete['delta_dist'] = calc_savings_kopt2(
            path_complete[['X', 'Y', 'X_next', 'Y_next', 'X_nei', 'Y_nei', 'X_nei_next', 'Y_nei_next']].values)

        # start constructing how penalizations go down by changes in position
        # list_prime = [i for i in primerange(0, cities.shape[0] + 1)]
        # path_df['isnotprime'] = 1
        # path_df.loc[list_prime, 'isnotprime'] = 0
        # path_df['mult10'] = 0
        # path_df.loc[path_df.position % 10 == 9, 'mult10'] = 1

        # path_df['penalization'] = 0.1 * path_df.dist * path_df.mult10.values * path_df['isnotprime'].values
        #
        # path_df['next_penalization'] = path_df['penalization'].rolling(10).sum().shift(-9)
        # path_df['last_penalization'] = path_df['penalization'].rolling(10).sum()
        #
        # path_df['last_prime'] = path_df['position'] - path_df['position'].where(path_df['isnotprime'] == 0).ffill()
        # path_df['next_prime'] = path_df['position'].where(path_df['isnotprime'] == 0).bfill() - path_df['position']

        # pasar datos de primos y penalizaciones  a path_complete y calcular beneficio
        # path_complete['last_prime_nei'] = path_df.loc[path_complete.neighbor, 'last_prime'].values
        # path_complete['last_pen'] = path_df.loc[path_complete.neighbor, 'last_penalization'].values
        # path_complete['willbe10'] = (path_complete['last_prime_nei'] + path_complete['position']) % 10
        # path_complete['delta_pen'] = (path_complete['willbe10'] == 0) * -path_complete['last_pen']

        # path_complete['final_savings'] = path_complete['delta_pen'] + path_complete['delta_dist']

        path_complete['swap_range'] = path_complete.nei_position - path_complete.position
        a = path_complete[(path_complete.swap_range > 200) & (path_complete.delta_dist < 4)]\
            .sort_values('swap_range', ascending=False)

        if a.shape[0] > 1000:
            a = a.iloc[:1000, :]
        score_pre = calc_score(np.append(path_df.index.values, 0), cities)
        score_min = score_pre
        for i in range(a.shape[0]):

            pos1 = int(a.iloc[i].position) + 1
            pos2 = int(a.iloc[i].nei_position)
            sub = np.append(path_df.index.values, 0).copy()

            sub[pos1:pos2] = sub[pos1:pos2][::-1]
            score_after = calc_score(sub, cities)
            # print(score_after - score_pre)
            if score_after < score_min:
                i_min = i
                score_min = score_after
                diff_score = score_after - score_pre
        if score_min == score_pre:
            break
        print('Score difference: {0:.2f}'.format(diff_score))
        pos1 = int(a.iloc[i_min].position) + 1
        pos2 = int(a.iloc[i_min].nei_position)
        sub = np.append(path_df.index.values, 0).copy()
        sub[pos1:pos2] = sub[pos1:pos2][::-1]
        score_after = calc_score(sub, cities)
        print('New score {0:.2f}'.format(score_after))
        scores.append(score_after)
    return np.append(path_df.index.values, 0).copy()


# # example of use (uncomment)
# x = time.time()
# cities = pd.read_csv('./input/cities.csv')
# sub = np_read_output_LKH('1129_2_LKH_local/output.txt')
# new_sub = opt_path_2opt(sub, cities)
# calc_score(new_sub, cities)
# print(str((time.time() -x)/60)+' minutes')
# pd.DataFrame({'Path': new_sub}).to_csv('1129_2_LKH_local/k_opt_sub.csv', index=False)
# new_subc = new_sub.copy()
# new_sub = new_subc.copy()



# 3: 2-opt optimization like this one: https://www.kaggle.com/kostyaatarik/close-ends-chunks-optimization-aka-2-opt

# NUMBA 2OPT



def numba_2opt(filename='1129_2_LKH_local/output.txt', path_out='',alpha=1):
    cities = pd.read_csv('input/cities.csv', index_col=['CityId'])
    XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
    is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)

    @numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
    def score_chunk(offset, chunk):
        pure_distance, penalty = 0.0, 0.0
        penalty_modulo = 9 - offset % 10
        for path_index in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[path_index], chunk[path_index+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
                penalty += distance
        return pure_distance + 0.1 * penalty

    @numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
    def cities_distance(offset, id_from, id_to):
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        if offset % 10 == 9 and is_not_prime[id_from]:
            return 1.1 * distance
        return distance

    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def score_path(path):
        return score_chunk(0, path)

    kdt = KDTree(XY)

    pairs = set()
    closest = []
    for city_id in tqdm(cities.index):
        dists, neibs = kdt.query([XY[city_id]], 31)
        closest.append(dists[0,1])
        for neib_id in neibs[0][1:]:
            if city_id and neib_id:  # skip pairs that include starting city
                pairs.add(tuple(sorted((city_id, neib_id))))
        neibs = kdt.query_radius([XY[city_id]], 31, count_only=False, return_distance=False)
        for neib_id in neibs[0]:
            if city_id and neib_id and city_id != neib_id:
                pairs.add(tuple(sorted((city_id, neib_id))))

    closest = np.array(closest)
    print(f'{len(pairs)} cities pairs are selected.')
    # sort pairs by distance
    pairs = np.array(list(pairs))
    distances = np.sum((XY[pairs.T[0]] - XY[pairs.T[1]])**2, axis=1)/((closest[pairs.T[0]] + closest[pairs.T[1]])**alpha)
    order = distances.argsort()
    pairs = pairs[order]

    path = np_read_output_LKH(filename)
    initial_score = score_path(path)

    path_index = np.argsort(path[:-1])

    total_score = initial_score

    print(f'Total score is {total_score:.2f}.')
    for _ in range(3):
        for step, (id1, id2) in enumerate(tqdm(pairs), 1):
            if step % 10 ** 6 == 0:
                new_total_score = score_path(path)
                print(
                    f'Score: {new_total_score:.2f}; improvement over last 10^6 steps: {total_score - new_total_score:.2f}; total improvement: {initial_score - new_total_score:.2f}.')
                total_score = new_total_score
            i, j = path_index[id1], path_index[id2]
            i, j = min(i, j), max(i, j)
            chunk, reversed_chunk = path[i - 1:j + 2], np.concatenate(
                [path[i - 1:i], path[j:i - 1:-1], path[j + 1:j + 2]])
            chunk_score, reversed_chunk_score = score_chunk(i - 1, chunk), score_chunk(i - 1, reversed_chunk)
            if j - i > 2:
                chunk_abc = np.concatenate([path[i - 1:i + 1], path[j:i:-1], path[j + 1:j + 2]])
                chunk_acb = np.concatenate([path[i - 1:i], path[j:j + 1], path[i:j], path[j + 1:j + 2]])
                chunk_abcb = np.concatenate([path[i - 1:i + 1], path[j:j + 1], path[i + 1:j], path[j + 1:j + 2]])
                abc_score, acb_score, abcb_score = map(lambda chunk: score_chunk(i - 1, chunk),
                                                       [chunk_abc, chunk_acb, chunk_abcb])
                for chunk, score, name in zip((chunk_abc, chunk_acb, chunk_abcb), (abc_score, acb_score, abcb_score),
                                              ('abc', 'acb', 'abcb')):
                    if score < chunk_score:
                        path[i - 1:j + 2] = chunk
                        path_index = np.argsort(path[:-1])  # update path index
                        chunk_score = score
            if reversed_chunk_score < chunk_score:
                path[i - 1:j + 2] = reversed_chunk
                path_index = np.argsort(path[:-1])  # update path index

    if path_out != '':
        assert(path_out != filename)
        pd.DataFrame({'Path': path}).to_csv(path_out + '_' + str(int(total_score)), index=False)

    return path


# numba_2opt(filename='1129_2_LKH_local/output.txt')
#
# numba_2opt(filename='1214_LKH_MC_local/output.txt', path_out='1214_LKH_MC_local/sub_2opt')
# numba_2opt(filename='1214_LKH_MC2_local/output.txt', path_out='1214_LKH_MC2_local/sub_2opt')
# numba_2opt(filename='1214_LKH_2_local/output.txt', path_out='1214_LKH_2_local/sub_2opt')
# numba_2opt(filename='1214_LKH_3_local/output.txt', path_out='1214_LKH_3_local/sub_2opt')
# numba_2opt(filename='1214_LKH_local/output.txt', path_out='1214_LKH_local/sub_2opt')
# numba_2opt(filename='1214_LKH_4_local/output.txt', path_out='1214_LKH_4_local/sub_2opt')
# numba_2opt(filename='1219_LKH_MT8_AWS/output.txt', path_out='1219_LKH_MT8_AWS/sub_2opt')

# numba_2opt(filename='1219_LKH_MT8_AWS/output.txt', path_out='1219_LKH_MT8_AWS/sub_2opt')
# numba_2opt(filename='1219_LKH_MT8init_AWS/output.txt', path_out='1219_LKH_MT8init_AWS/sub_2opt')
# numba_2opt(filename='1222_LKH_PA2C1_init_AWS/output.txt', path_out='1222_LKH_PA2C1_init_AWS/sub_2opt')
# numba_2opt(filename='1222_LKH_PA2C1_AWS/output.txt', path_out='1222_LKH_PA2C1_AWS/sub_2opt')
# numba_2opt(filename='1222_LKH_MT10_init_AWS/output.txt', path_out='1222_LKH_MT10_init_AWS/sub_2opt')
# numba_2opt(filename='1222_LKH_A4C3_init_local/output.txt', path_out='1222_LKH_A4C3_init_local/sub_2opt')



# 4: Optimizations getting 3 cities at a time and trying to permutate the order

def numba_3comb(filename='1129_2_LKH_local/output.txt', path_out='', alpha=1):

    cities = pd.read_csv('input/cities.csv', index_col=['CityId'])
    XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
    is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)


    @numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
    def cities_distance(offset, id_from, id_to):
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        if offset % 10 == 9 and is_not_prime[id_from]:
            return 1.1 * distance
        return distance


    @numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
    def score_chunk(offset, chunk):
        pure_distance, penalty = 0.0, 0.0
        penalty_modulo = 9 - offset % 10
        for path_index in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[path_index], chunk[path_index+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
                penalty += distance
        return pure_distance + 0.1 * penalty


    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def score_path(path):
        return score_chunk(0, path)


    @numba.jit
    def chunk_scores(chunk):
        scores = np.zeros(10)
        pure_distance = 0
        for i in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[i], chunk[i+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if is_not_prime[id_from]:
                scores[9-i%10] += distance
        scores *= 0.1
        scores += pure_distance
        return scores

    def score_compound_chunk(offset, head, chunks, tail, scores, indexes_permutation=None):
        if indexes_permutation is None:
            indexes_permutation = range(len(chunks))
        score = 0.0
        last_city_id = head
        for index in indexes_permutation:
            chunk, chunk_scores = chunks[index], scores[index]
            score += cities_distance(offset % 10, last_city_id, chunk[0])
            score += chunk_scores[(offset + 1) % 10]
            last_city_id = chunk[-1]
            offset += len(chunk)
        return score + cities_distance(offset % 10, last_city_id, tail)


    kdt = KDTree(XY)


    triplets = set()
    closest = []
    for city_id in tqdm(cities.index):
        dists, neibs = kdt.query([XY[city_id]], 9)
        closest.append(dists[0, 1])
        for triplet in combinations(neibs[0], 3):
            if all(triplet):
                triplets.add(tuple(sorted(triplet)))
        neibs = kdt.query_radius([XY[city_id]], 10, count_only=False, return_distance=False)
        for triplet in combinations(neibs[0], 3):
            if all(triplet):
                triplets.add(tuple(sorted(triplet)))

    closest = np.array(closest)
    print(f'{len(triplets)} cities triplets are selected.')

    # sort triplets by distance
    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def sum_distance(ids):
        res = 0
        for i in numba.prange(len(ids)):
            for j in numba.prange(i + 1, len(ids)):
                res += cities_distance(0, ids[i], ids[j])
        return res

    triplets = np.array(list(triplets))
    distances = np.array(list(map(sum_distance, tqdm(triplets))))/((closest[triplets.T[0]] + closest[triplets.T[1]] + closest[triplets.T[2]])**alpha)
    order = distances.argsort()
    triplets = triplets[order]


    path = pd.read_csv(filename).Path.values


    def not_trivial_permutations(iterable):
        perms = permutations(iterable)
        next(perms)
        yield from perms


    @lru_cache(maxsize=None)
    def not_trivial_indexes_permutations(length):
        return np.array([list(p) for p in not_trivial_permutations(range(length))])

    path_index = np.argsort(path[:-1])
    print(f'Total score is {score_path(path):.2f}.')
    for _ in range(3):
        for ids in tqdm(triplets):
            i, j, k = sorted(path_index[ids])
            head, tail = path[i-1], path[k+1]
            chunks = [path[i:i+1], path[i+1:j], path[j:j+1], path[j+1:k], path[k:k+1]]
            chunks = [chunk for chunk in chunks if len(chunk)]
            scores = [chunk_scores(chunk) for chunk in chunks]
            default_score = score_compound_chunk(i-1, head, chunks, tail, scores)
            best_score = default_score
            for indexes_permutation in not_trivial_indexes_permutations(len(chunks)):
                score = score_compound_chunk(i-1, head, chunks, tail, scores, indexes_permutation)
                if score < best_score:
                    permutation = [chunks[i] for i in indexes_permutation]
                    best_chunk = np.concatenate([[head], np.concatenate(permutation), [tail]])
                    best_score = score
            if best_score < default_score:
                path[i-1:k+2] = best_chunk
                path_index = np.argsort(path[:-1])
                print(f'New total score is {score_path(path):.2f}. Permutating path at indexes {i}, {j}, {k}.')
        triplets = triplets[:10**6]

    best_score = score_path(path)
    if path_out != '':
        assert(path_out != filename)
        pd.DataFrame({'Path': path}).to_csv(path_out + '_' + str(int(best_score)), index=False)

# numba_3comb(filename='1219_LKH_MT8_AWS/sub_2opt_1516029', path_out='1219_LKH_MT8_AWS/3comb')
# numba_3comb(filename='1219_LKH_MT8init_AWS/sub_2opt_1516101', path_out='1219_LKH_MT8init_AWS/3comb')
# numba_3comb(filename='1222_LKH_PA2C1_init_AWS/sub_2opt_1516079', path_out='1222_LKH_PA2C1_init_AWS/3comb')
# numba_3comb(filename='1222_LKH_PA2C1_AWS/sub_2opt_1516089', path_out='1222_LKH_PA2C1_AWS/3comb')
# numba_3comb(filename='1222_LKH_MT10_init_AWS/sub_2opt_1516036', path_out='1222_LKH_MT10_init_AWS/3comb')
# numba_3comb(filename='1222_LKH_A4C3_init_local/sub_2opt_1516018', path_out='1222_LKH_A4C3_init_local/3comb')



# 5: Optimizations dividing current path into chunks of 500 and runing LKH again with the difference that we add
# penalties to the paths that are number 10 so that we find a different path


# LKH to mini paths
def subtram_lkh(file_path='1222_LKH_MT10_init_AWS/3comb_1515779', path_out='', max_i=500, tempdir='temp',
                lkh_path = '/home/luis/src'):
    # final_results = []pen_list
    x = time.time()
    cities = pd.read_csv('./input/cities.csv')
    np_xy_cities = cities[['X', 'Y']].values

    sub = pd.read_csv(file_path).Path.values

    old_score = calc_score(sub, cities)
    print('old score: ' + str(old_score)) #, flush = True)
    dists, pens = calc_2_vec_score(sub, cities)

    for i in tqdm(range(max_i)):
        sub_size = 500
        max_pos = int((sub.shape[0] - sub_size)*i/max_i)

        subtour = sub[max_pos:(max_pos + sub_size)]
        sub_c = sub.copy()
        # results = {'multiplier': [0], 'score_loops': [old_score]}
        best_multiplier = 0
        old_score_2 = calc_score(sub, cities)
        best_score = old_score_2
        multipliers = np.exp(np.arange(-12, 1) / 4).tolist()
        for loop, multiplier in enumerate(multipliers):
            subpens = pens[max_pos:(max_pos + sub_size)]
            pen_pos = np.where(subpens > 0)
            pen_value = subpens[pen_pos] * multiplier
            pen_list = [pen_pos[0], pen_value]

            parameters = [
                ("PROBLEM_FILE", tempdir + "/mat_.tsp"),
                ("OUTPUT_TOUR_FILE", tempdir + "/tsp_solution.csv"),
                ("SEED", 2018),
                ('CANDIDATE_SET_TYPE', 'POPMUSIC'),  # 'NEAREST-NEIGHBOR', 'ALPHA'),
            ]
            write_parameters_temp(parameters, filename=tempdir + '/mat_.par')
            write_tsp_file_temp(subtour, np_xy_cities, filename=tempdir + '/mat_.tsp', name='ts', pen_list=pen_list)

            os.system(lkh_path + '/LKH-2.0.9/LKH ' + tempdir + '/mat_.par > /dev/null')

            path = np_read_output_LKH(tempdir + '/tsp_solution.csv')[:-1]
            if path[499] == 499:
                # print('mean dev of new path: ' + str(np.std(path[1:]-path[:-1])))
                path = subtour[path]
                sub_c[max_pos:(max_pos + sub_size)] = path
                score_loop = calc_score(sub_c, cities)
                if score_loop < best_score:
                    best_score = score_loop
                    best_multiplier = multiplier

        if best_score < old_score_2:
            print("saving: " + str(old_score_2 - best_score) + ' ... ' + 'current_score: ' + str(int(best_score)))
            subpens = pens[max_pos:(max_pos + sub_size)]
            pen_pos = np.where(subpens > 0)
            pen_value = subpens[pen_pos] * best_multiplier
            pen_list = [pen_pos[0], pen_value]

            parameters = [
                ("PROBLEM_FILE", tempdir + "/mat_.tsp"),
                ("OUTPUT_TOUR_FILE", tempdir + "/tsp_solution.csv"),
                ("SEED", 2018),
                ('CANDIDATE_SET_TYPE', 'POPMUSIC'),  # 'NEAREST-NEIGHBOR', 'ALPHA'),
            ]
            write_parameters_temp(parameters,  filename=tempdir + '/mat_.par')
            write_tsp_file_temp(subtour, np_xy_cities, filename=tempdir + '/mat_.tsp', name='ts', pen_list=pen_list)

            os.system(lkh_path + '/LKH-2.0.9/LKH ' + tempdir + '/mat_.par > /dev/null')

            path = np_read_output_LKH(tempdir + '/tsp_solution.csv')[:-1]
            assert(path[499]==499)
            # print('mean dev of new path: ' + str(np.std(path[1:]-path[:-1])))
            path = subtour[path]
            sub_c[max_pos:(max_pos + sub_size)] = path
            score_loop = calc_score(sub_c, cities)
            # results['multiplier'].append(100)
            # results['score_loops'].append(score_loop)
            sub = sub_c.copy()

        # results = pd.DataFrame(results)
        # results['diff_score'] = results['score_loops'] - results['score_loops'][0]
        # results['try'] = i
        # final_results.append(results.copy())

    final_score = calc_score(sub, cities)
    print("saving: " + str(old_score - best_score))
    print("final score: " + str(final_score))
    print(str((time.time() - x) / 60) + ' minutes')

    # final_results = pd.concat(final_results)
    # final_results['under0'] = final_results['diff_score'] < 0
    # under0_mean = final_results.groupby('multiplier')['under0'].mean()
    # under0_exec = final_results.groupby('try')['under0'].mean()
    # print(final_results.groupby('try')['diff_score'].min().mean() / subtour.shape[0] * sub.shape[0])

    pd.DataFrame({'Path': sub}).to_csv(path_out + '500opt_' + str(int(final_score)), index=False)

# subtram_lkh(file_path='1214_LKH_MC2_local/sub_2opt_1516258', path_out = '1214_LKH_MC2_local/')
# subtram_lkh(file_path='1219_LKH_MT8_AWS/sub_2opt_1516029', path_out='1219_LKH_MT8_AWS/sub_', tempdir='temp')
# subtram_lkh(file_path='1222_LKH_PA2C1_AWS/sub_2opt_1516089', path_out='1222_LKH_PA2C1_AWS/sub_', tempdir='temp2')
# subtram_lkh(file_path='1219_LKH_MT8init_AWS/sub_2opt_1516101', path_out='1219_LKH_MT8init_AWS/sub_', tempdir='temp3')
# subtram_lkh(file_path='1222_LKH_MT10_init_AWS/sub_2opt_1516036', path_out='1222_LKH_MT10_init_AWS/sub_', tempdir='temp4')
# subtram_lkh(file_path='1222_LKH_PA2C1_init_AWS/sub_2opt_1516079', path_out='1222_LKH_PA2C1_init_AWS/sub_', tempdir='temp5')
# subtram_lkh(file_path='1222_LKH_A4C3_init_local/sub_2opt_1516018', path_out='1222_LKH_A4C3_init_local/sub_', tempdir='temp6')

# 6: Optimizations dividing current path into chunks of 500 and runing LKH again with the difference that we add
# penalties to the paths that are number 10 so that we find a different path


# LKH to mini paths
def subtram_lkh_rec(file_path='1214_LKH_MC2_local/sub_2opt_1516258', path_out='', max_i=500, tempdir='temp2', multiply = 1.2,
                    lkh_path = '/home/luis/src'):
    # final_results = []pen_list
    x = time.time()
    cities = pd.read_csv('./input/cities.csv')
    np_xy_cities = cities[['X', 'Y']].values

    sub = pd.read_csv(file_path).Path.values

    old_score = calc_score(sub, cities)
    print('old score: ' + str(old_score))
    dists, pens = calc_2_vec_score(sub, cities)

    for i in tqdm(range(max_i)):
        sub_size = 500
        max_pos = int((sub.shape[0] - sub_size)*i/max_i)

        subtour = sub[max_pos:(max_pos + sub_size)]
        sub_c = sub.copy()

        old_score_2 = calc_score(sub, cities)
        best_score = old_score_2

        subpens = pens[max_pos:(max_pos + sub_size - 1)]

        pen_matrix = np.zeros([sub_size, sub_size])
        for j in range(max_i-1):
           pen_matrix[j,j+1] = subpens[j]
        # print('mean dev of new path: ' + str(np.std(subtour[1:] - subtour[:-1])) + ' ... ' + str(old_score_2))
        # print(pen_matrix[85, 86])
        for loop in range(20):

            parameters = [
                ("PROBLEM_FILE", tempdir + "/mat_.tsp"),
                ("OUTPUT_TOUR_FILE", tempdir + "/tsp_solution.csv"),
                ("SEED", 2018),
                ('CANDIDATE_SET_TYPE', 'POPMUSIC'),  # 'NEAREST-NEIGHBOR', 'ALPHA'),
            ]
            write_parameters_temp(parameters, filename=tempdir + '/mat_.par')
            write_tsp_file_temp2(subtour, np_xy_cities, filename=tempdir + '/mat_.tsp', name='ts', pen_matrix=pen_matrix)

            os.system(lkh_path + '/LKH-2.0.9/LKH ' + tempdir + '/mat_.par > /dev/null')

            path_ = np_read_output_LKH(tempdir + '/tsp_solution.csv')[:-1]

            path = subtour[path_]
            sub_c[max_pos:(max_pos + sub_size)] = path
            score_loop = calc_score(sub_c, cities)
            # print('mean dev of new path: ' + str(np.std(path[1:] - path[:-1])) + ' ... ' + str(score_loop))

            dists_, pens_ = calc_2_vec_score(sub_c, cities)
            subpens_ = pens_[max_pos:(max_pos + sub_size - 1)]

            pen_matrix_new = np.zeros([sub_size, sub_size])
            for j in range(max_i - 1):
                if subpens_[j] > 0:
                    max_j = max(path_[j], path_[j + 1])
                    min_j = min(path_[j], path_[j + 1])
                    pen_matrix_new[min_j, max_j] = subpens_[j]

            pen_matrix = (pen_matrix * (loop + 1) / (loop + 2) + pen_matrix_new * (1 / (loop + 2)))*multiply
            # print(pen_matrix[85, 86])
            if score_loop < best_score:
                best_score = score_loop
                best_sub = sub_c.copy()
                # best_multiplier = multiplier

        if best_score < old_score_2:
            print("saving: " + str(old_score_2 - best_score) + ' ... ' + 'current_score: ' + str(int(best_score)))
            sub = best_sub.copy()

    final_score = calc_score(sub, cities)
    print("saving: " + str(old_score - best_score))
    print("final score: " + str(final_score))
    print(str((time.time() - x) / 60) + ' minutes')

    # final_results = pd.concat(final_results)
    # final_results['under0'] = final_results['diff_score'] < 0
    # under0_mean = final_results.groupby('multiplier')['under0'].mean()
    # under0_exec = final_results.groupby('try')['under0'].mean()
    # print(final_results.groupby('try')['diff_score'].min().mean() / subtour.shape[0] * sub.shape[0])

    pd.DataFrame({'Path': sub}).to_csv(path_out + '501opt_' + str(int(final_score)), index=False)


def numba_4comb(filename='1219_LKH_MT8_AWS/sub_2opt_1516029', path_out=''):

    cities = pd.read_csv('input/cities.csv', index_col=['CityId'])
    XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
    is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)


    @numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
    def cities_distance(offset, id_from, id_to):
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        if offset % 10 == 9 and is_not_prime[id_from]:
            return 1.1 * distance
        return distance


    @numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
    def score_chunk(offset, chunk):
        pure_distance, penalty = 0.0, 0.0
        penalty_modulo = 9 - offset % 10
        for path_index in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[path_index], chunk[path_index+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
                penalty += distance
        return pure_distance + 0.1 * penalty


    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def score_path(path):
        return score_chunk(0, path)


    @numba.jit
    def chunk_scores(chunk):
        scores = np.zeros(10)
        pure_distance = 0
        for i in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[i], chunk[i+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if is_not_prime[id_from]:
                scores[9-i%10] += distance
        scores *= 0.1
        scores += pure_distance
        return scores

    def score_compound_chunk(offset, head, chunks, tail, scores, indexes_permutation=None):
        if indexes_permutation is None:
            indexes_permutation = range(len(chunks))
        score = 0.0
        last_city_id = head
        for index in indexes_permutation:
            chunk, chunk_scores = chunks[index], scores[index]
            score += cities_distance(offset % 10, last_city_id, chunk[0])
            score += chunk_scores[(offset + 1) % 10]
            last_city_id = chunk[-1]
            offset += len(chunk)
        return score + cities_distance(offset % 10, last_city_id, tail)


    kdt = KDTree(XY)


    path = pd.read_csv(filename).Path.values
    path_index = np.argsort(path[:-1])


    triplets = set()
    for city_id in tqdm(cities.index):
        dists, neibs = kdt.query([XY[city_id]], 9)
        for triplet in combinations(neibs[0], 4):
            if all(triplet):
                if max(path_index[list(triplet)]) - min(path_index[list(triplet)]) > 10:
                    triplets.add(tuple(sorted(triplet)))
        neibs = kdt.query_radius([XY[city_id]], 10, count_only=False, return_distance=False)
        for triplet in combinations(neibs[0], 4):
            if all(triplet):
                if max(path_index[list(triplet)]) - min(path_index[list(triplet)]) > 10:
                    triplets.add(tuple(sorted(triplet)))

    print(f'{len(triplets)} cities triplets are selected.')

    # sort triplets by distance
    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def sum_distance(ids):
        res = 0
        for i in numba.prange(len(ids)):
            for j in numba.prange(i + 1, len(ids)):
                res += cities_distance(0, ids[i], ids[j])
        return res

    triplets = np.array(list(triplets))
    distances = np.array(list(map(sum_distance, tqdm(triplets))))
    order = distances.argsort()
    triplets = triplets[order]


    path = pd.read_csv(filename).Path.values


    def not_trivial_permutations(iterable):
        perms = permutations(iterable)
        next(perms)
        yield from perms


    @lru_cache(maxsize=None)
    def not_trivial_indexes_permutations(length):
        return np.array([list(p) for p in not_trivial_permutations(range(length))])

    path_index = np.argsort(path[:-1])
    print(f'Total score is {score_path(path):.2f}.')
    for _ in range(2):
        for ids in tqdm(triplets):
            i, j, k, l = sorted(path_index[ids])
            head, tail = path[i-1], path[l+1]
            # chunks = [path[i:i+1], path[i+1:j], path[j:j+1], path[j+1:k], path[k:k+1], path[k+1:l], path[l:l+1]]
            chunks = [path[i:j], path[j:k], path[k:l], path[l:l + 1]]
            # chunks = [chunk for chunk in chunks if len(chunk)]
            scores = [chunk_scores(chunk) for chunk in chunks]
            default_score = score_compound_chunk(i-1, head, chunks, tail, scores)
            best_score = default_score
            for indexes_permutation in not_trivial_indexes_permutations(len(chunks)):
                score = score_compound_chunk(i-1, head, chunks, tail, scores, indexes_permutation)
                if score < best_score:
                    permutation = [chunks[i] for i in indexes_permutation]
                    best_chunk = np.concatenate([[head], np.concatenate(permutation), [tail]])
                    best_score = score
            if best_score < default_score:
                path[i-1:l+2] = best_chunk
                path_index = np.argsort(path[:-1])
                print(f'New total score is {score_path(path):.2f}. Permutating path at indexes {i}, {j}, {k}, {l}.')
        triplets = triplets[:10**6]

    best_score = score_path(path)
    if path_out != '':
        assert(path_out != filename)
        pd.DataFrame({'Path': path}).to_csv(path_out + '_' + str(int(best_score)), index=False)

# numba_4comb(filename='1219_LKH_MT8_AWS/sub_2opt_1516029', path_out='1219_LKH_MT8_AWS/3comb')


def numba_5comb(filename='1219_LKH_MT8_AWS/sub_2opt_1516029', path_out=''):

    cities = pd.read_csv('input/cities.csv', index_col=['CityId'])
    XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
    is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)


    @numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
    def cities_distance(offset, id_from, id_to):
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        if offset % 10 == 9 and is_not_prime[id_from]:
            return 1.1 * distance
        return distance


    @numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
    def score_chunk(offset, chunk):
        pure_distance, penalty = 0.0, 0.0
        penalty_modulo = 9 - offset % 10
        for path_index in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[path_index], chunk[path_index+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
                penalty += distance
        return pure_distance + 0.1 * penalty


    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def score_path(path):
        return score_chunk(0, path)


    @numba.jit
    def chunk_scores(chunk):
        scores = np.zeros(10)
        pure_distance = 0
        for i in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[i], chunk[i+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if is_not_prime[id_from]:
                scores[9-i%10] += distance
        scores *= 0.1
        scores += pure_distance
        return scores

    def score_compound_chunk(offset, head, chunks, tail, scores, indexes_permutation=None):
        if indexes_permutation is None:
            indexes_permutation = range(len(chunks))
        score = 0.0
        last_city_id = head
        for index in indexes_permutation:
            chunk, chunk_scores = chunks[index], scores[index]
            score += cities_distance(offset % 10, last_city_id, chunk[0])
            score += chunk_scores[(offset + 1) % 10]
            last_city_id = chunk[-1]
            offset += len(chunk)
        return score + cities_distance(offset % 10, last_city_id, tail)


    kdt = KDTree(XY)


    path = pd.read_csv(filename).Path.values
    path_index = np.argsort(path[:-1])


    triplets = set()
    closest = []
    for city_id in tqdm(cities.index):
        dists, neibs = kdt.query([XY[city_id]], 9)
        closest.append(dists[0, 1])
        for triplet in combinations(neibs[0], 5):
            if all(triplet):
                if max(path_index[list(triplet)]) - min(path_index[list(triplet)]) > 15:
                    triplets.add(tuple(sorted(triplet)))
        neibs = kdt.query_radius([XY[city_id]], 10, count_only=False, return_distance=False)
        for triplet in combinations(neibs[0], 5):
            if all(triplet):
                if max(path_index[list(triplet)]) - min(path_index[list(triplet)]) > 1:
                    triplets.add(tuple(sorted(triplet)))

    print(f'{len(triplets)} cities triplets are selected.')

         # sort triplets by distance

    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def sum_distance(ids):
        res = 0
        for i in numba.prange(len(ids)):
            for j in numba.prange(i + 1, len(ids)):
                res += cities_distance(0, ids[i], ids[j])
        return res

    triplets = np.array(list(triplets))
    closest = np.array(closest)
    distances = np.array(list(map(sum_distance, tqdm(triplets))))
    distances = distances / (closest[triplets.T[0]] + closest[triplets.T[1]] + closest[triplets.T[2]])
    order = distances.argsort()
    triplets = triplets[order]

    path = pd.read_csv(filename).Path.values

    def not_trivial_permutations(iterable):
        perms = permutations(iterable)
        next(perms)
        yield from perms

    @lru_cache(maxsize=None)
    def not_trivial_indexes_permutations(length):
        return np.array([list(p) for p in not_trivial_permutations(range(length))
                         # if (p[0] != 0) and (p[1] != 1) and (p[2] != 2) and (p[3] != 3) and (p[4] != 4)])
            if (p[1]-p[0] != 1) and (p[2]-p[1] != 1) and (p[3]-p[2] != 1) and (p[4]-p[3] != 1) and (p[4] != 4) and (p[0] != 0)])
    path_index = np.argsort(path[:-1])
    print(f'Total score is {score_path(path):.2f}.')
    for _ in range(2):
        for ids in tqdm(triplets):
            i, j, k, l, m = sorted(path_index[ids])
            head, tail = path[i-1], path[m+1]
            # chunks = [path[i:i+1], path[i+1:j], path[j:j+1], path[j+1:k], path[k:k+1], path[k+1:l], path[l:l+1]]
            chunks = [path[i:j], path[j:k], path[k:l], path[l:m], path[m:m + 1]]
            # chunks = [chunk for chunk in chunks if len(chunk)]
            scores = [chunk_scores(chunk) for chunk in chunks]
            default_score = score_compound_chunk(i-1, head, chunks, tail, scores)
            best_score = default_score
            for indexes_permutation in not_trivial_indexes_permutations(len(chunks)):
                score = score_compound_chunk(i-1, head, chunks, tail, scores, indexes_permutation)
                if score < best_score:
                    permutation = [chunks[i] for i in indexes_permutation]
                    best_chunk = np.concatenate([[head], np.concatenate(permutation), [tail]])
                    best_score = score
            if best_score < default_score:
                path[i-1:m+2] = best_chunk
                path_index = np.argsort(path[:-1])
                print(f'New total score is {score_path(path):.2f}. Permutating path at indexes {i}, {j}, {k}, {l}, {m}.')
        triplets = triplets[:10**6]

    best_score = score_path(path)
    if path_out != '':
        assert(path_out != filename)
        pd.DataFrame({'Path': path}).to_csv(path_out + '_' + str(int(best_score)), index=False)

# numba_5comb(filename='1219_LKH_MT8_AWS/sub_2opt_1516029', path_out='1219_LKH_MT8_AWS/3comb')

def numba_5comb2(filename='1219_LKH_MT8_AWS/sub_2opt_1516029', path_out=''):

    cities = pd.read_csv('input/cities.csv', index_col=['CityId'])
    XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
    is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)

    @numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
    def cities_distance(offset, id_from, id_to):
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        if offset % 10 == 9 and is_not_prime[id_from]:
            return 1.1 * distance
        return distance


    @numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
    def score_chunk(offset, chunk):
        pure_distance, penalty = 0.0, 0.0
        penalty_modulo = 9 - offset % 10
        for path_index in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[path_index], chunk[path_index+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
                penalty += distance
        return pure_distance + 0.1 * penalty


    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def score_path(path):
        return score_chunk(0, path)


    @numba.jit
    def chunk_scores(chunk):
        scores = np.zeros(10)
        pure_distance = 0
        for i in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[i], chunk[i+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if is_not_prime[id_from]:
                scores[9-i%10] += distance
        scores *= 0.1
        scores += pure_distance
        return scores


    @numba.jit('f8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:])', nopython=True, parallel=False)
    def score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes):
        score = 0.0
        last_city_id = head
        for i in numba.prange(len(indexes)):
            index = indexes[i]
            first, last, chunk_len = firsts[index], lasts[index], lens[index]
            score += cities_distance(offset, last_city_id, first)
            score += scores[index, (offset + 1) % 10]
            last_city_id = last
            offset += chunk_len
        return score + cities_distance(offset, last_city_id, tail)


    @numba.jit('i8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:,:], f8)', nopython=True, parallel=False)
    def best_score_permutation_index(offset, head, firsts, lasts, lens, tail, scores, indexes, best_score):
        best_index = -1
        for i in numba.prange(len(indexes)):
            score = score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes[i])
            if score < best_score:
                best_index, best_score = i, score
        return best_index


    kdt = KDTree(XY)

    fives = set()
    closest = []
    for i in tqdm(cities.index):
        dists, neibs = kdt.query([XY[i]], 9)
        closest.append(dists[0, 1])
        for comb in combinations(neibs[0], 5):
            if all(comb):
                fives.add(tuple(sorted(comb)))
        neibs = kdt.query_radius([XY[i]], 10, count_only=False, return_distance=False)
        for comb in combinations(neibs[0], 5):
            if all(comb):
                fives.add(tuple(sorted(comb)))

    print(f'{len(fives)} cities fives are selected.')


    # sort fives by distance
    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def sum_distance(ids):
        res = 0
        for i in numba.prange(len(ids)):
            for j in numba.prange(i + 1, len(ids)):
                res += cities_distance(0, ids[i], ids[j])
        return res


    fives = np.array(list(fives))
    closest = np.array(closest)
    distances = np.array(list(map(sum_distance, tqdm(fives))))
    distances = distances / (closest[fives.T[0]] + closest[fives.T[1]] + closest[fives.T[2]] + closest[fives.T[3]] + closest[fives.T[4]])
    order = distances.argsort()
    fives = fives[order]
    # Sergi 2019-01-08 20:50:00 Lo cambio para ver si mejora
    #fives = fives[:2 * 10 ** 6]
    #fives = fives[2 * 10 ** 6:4 * 10 ** 6]
    fives = fives[2 * 10 ** 6:5 * 10 ** 6]

    import gc
    gc.collect()

    path = pd.read_csv(filename).Path.values


    @lru_cache(maxsize=None)
    def indexes_permutations(n):
        return np.array(list(map(list, permutations(range(n)))))


    path_index = np.argsort(path[:-1])
    print(f'Total score is {score_path(path):.2f}.')
    for _ in range(2):
        for ids in tqdm(fives):
            i1, i2, i3, i4, i5 = np.sort(path_index[ids])
            head, tail = path[i1-1], path[i5+1]
            chunks = [path[i1:i1+1], path[i1+1:i2], path[i2:i2+1], path[i2+1:i3],
                      path[i3:i3+1], path[i3+1:i4], path[i4:i4+1], path[i4+1:i5], path[i5:i5+1]]
            chunks = [chunk for chunk in chunks if len(chunk)]
            scores = np.array([chunk_scores(chunk) for chunk in chunks])
            lens = np.array([len(chunk) for chunk in chunks])
            firsts = np.array([chunk[0] for chunk in chunks])
            lasts = np.array([chunk[-1] for chunk in chunks])
            best_score = score_compound_chunk(i1-1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks))[0])
            index = best_score_permutation_index(i1-1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks)), best_score)
            if index > 0:
                perm = [chunks[i] for i in indexes_permutations(len(chunks))[index]]
                path[i1-1:i5+2] = np.concatenate([[head], np.concatenate(perm), [tail]])
                path_index = np.argsort(path[:-1])
                print(f'New total score is {score_path(path):.3f}. Permutating path at indexes {i1}, {i2}, {i3}, {i4}, {i5}.')

    best_score = score_path(path)
    if path_out != '':
        assert (path_out != filename)
        pd.DataFrame({'Path': path}).to_csv(path_out + '_' + str(int(best_score)), index=False)

def numba_5comb2_8M(filename='1219_LKH_MT8_AWS/sub_2opt_1516029', path_out=''):

    cities = pd.read_csv('input/cities.csv', index_col=['CityId'])
    XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
    is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)

    @numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
    def cities_distance(offset, id_from, id_to):
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        if offset % 10 == 9 and is_not_prime[id_from]:
            return 1.1 * distance
        return distance


    @numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
    def score_chunk(offset, chunk):
        pure_distance, penalty = 0.0, 0.0
        penalty_modulo = 9 - offset % 10
        for path_index in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[path_index], chunk[path_index+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
                penalty += distance
        return pure_distance + 0.1 * penalty


    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def score_path(path):
        return score_chunk(0, path)


    @numba.jit
    def chunk_scores(chunk):
        scores = np.zeros(10)
        pure_distance = 0
        for i in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[i], chunk[i+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if is_not_prime[id_from]:
                scores[9-i%10] += distance
        scores *= 0.1
        scores += pure_distance
        return scores


    @numba.jit('f8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:])', nopython=True, parallel=False)
    def score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes):
        score = 0.0
        last_city_id = head
        for i in numba.prange(len(indexes)):
            index = indexes[i]
            first, last, chunk_len = firsts[index], lasts[index], lens[index]
            score += cities_distance(offset, last_city_id, first)
            score += scores[index, (offset + 1) % 10]
            last_city_id = last
            offset += chunk_len
        return score + cities_distance(offset, last_city_id, tail)


    @numba.jit('i8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:,:], f8)', nopython=True, parallel=False)
    def best_score_permutation_index(offset, head, firsts, lasts, lens, tail, scores, indexes, best_score):
        best_index = -1
        for i in numba.prange(len(indexes)):
            score = score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes[i])
            if score < best_score:
                best_index, best_score = i, score
        return best_index


    kdt = KDTree(XY)

    fives = set()
    closest = []
    for i in tqdm(cities.index):
        dists, neibs = kdt.query([XY[i]], 9)
        closest.append(dists[0, 1])
        for comb in combinations(neibs[0], 5):
            if all(comb):
                fives.add(tuple(sorted(comb)))
        neibs = kdt.query_radius([XY[i]], 10, count_only=False, return_distance=False)
        for comb in combinations(neibs[0], 5):
            if all(comb):
                fives.add(tuple(sorted(comb)))

    print(f'{len(fives)} cities fives are selected.')


    # sort fives by distance
    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def sum_distance(ids):
        res = 0
        for i in numba.prange(len(ids)):
            for j in numba.prange(i + 1, len(ids)):
                res += cities_distance(0, ids[i], ids[j])
        return res


    fives = np.array(list(fives))
    closest = np.array(closest)
    distances = np.array(list(map(sum_distance, tqdm(fives))))
    distances = distances / (closest[fives.T[0]] + closest[fives.T[1]] + closest[fives.T[2]] + closest[fives.T[3]] + closest[fives.T[4]])
    order = distances.argsort()
    fives = fives[order]
    # Sergi 2019-01-08 20:50:00 Lo cambio para ver si mejora
    #fives = fives[:2 * 10 ** 6]
    #fives = fives[2 * 10 ** 6:4 * 10 ** 6]
    #fives = fives[2 * 10 ** 6:5 * 10 ** 6]
    fives = fives[5 * 10 ** 6:8 * 10 ** 6]

    import gc
    gc.collect()

    path = pd.read_csv(filename).Path.values


    @lru_cache(maxsize=None)
    def indexes_permutations(n):
        return np.array(list(map(list, permutations(range(n)))))


    path_index = np.argsort(path[:-1])
    print(f'Total score is {score_path(path):.2f}.')
    for _ in range(2):
        for ids in tqdm(fives):
            i1, i2, i3, i4, i5 = np.sort(path_index[ids])
            head, tail = path[i1-1], path[i5+1]
            chunks = [path[i1:i1+1], path[i1+1:i2], path[i2:i2+1], path[i2+1:i3],
                      path[i3:i3+1], path[i3+1:i4], path[i4:i4+1], path[i4+1:i5], path[i5:i5+1]]
            chunks = [chunk for chunk in chunks if len(chunk)]
            scores = np.array([chunk_scores(chunk) for chunk in chunks])
            lens = np.array([len(chunk) for chunk in chunks])
            firsts = np.array([chunk[0] for chunk in chunks])
            lasts = np.array([chunk[-1] for chunk in chunks])
            best_score = score_compound_chunk(i1-1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks))[0])
            index = best_score_permutation_index(i1-1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks)), best_score)
            if index > 0:
                perm = [chunks[i] for i in indexes_permutations(len(chunks))[index]]
                path[i1-1:i5+2] = np.concatenate([[head], np.concatenate(perm), [tail]])
                path_index = np.argsort(path[:-1])
                print(f'New total score is {score_path(path):.3f}. Permutating path at indexes {i1}, {i2}, {i3}, {i4}, {i5}.')

    best_score = score_path(path)
    if path_out != '':
        assert (path_out != filename)
        pd.DataFrame({'Path': path}).to_csv(path_out + '_' + str(int(best_score)), index=False)

def numba_5comb2_28M(filename='1219_LKH_MT8_AWS/sub_2opt_1516029', path_out=''):

    cities = pd.read_csv('input/cities.csv', index_col=['CityId'])
    XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
    is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)

    @numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
    def cities_distance(offset, id_from, id_to):
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        if offset % 10 == 9 and is_not_prime[id_from]:
            return 1.1 * distance
        return distance


    @numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
    def score_chunk(offset, chunk):
        pure_distance, penalty = 0.0, 0.0
        penalty_modulo = 9 - offset % 10
        for path_index in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[path_index], chunk[path_index+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
                penalty += distance
        return pure_distance + 0.1 * penalty


    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def score_path(path):
        return score_chunk(0, path)


    @numba.jit
    def chunk_scores(chunk):
        scores = np.zeros(10)
        pure_distance = 0
        for i in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[i], chunk[i+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if is_not_prime[id_from]:
                scores[9-i%10] += distance
        scores *= 0.1
        scores += pure_distance
        return scores


    @numba.jit('f8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:])', nopython=True, parallel=False)
    def score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes):
        score = 0.0
        last_city_id = head
        for i in numba.prange(len(indexes)):
            index = indexes[i]
            first, last, chunk_len = firsts[index], lasts[index], lens[index]
            score += cities_distance(offset, last_city_id, first)
            score += scores[index, (offset + 1) % 10]
            last_city_id = last
            offset += chunk_len
        return score + cities_distance(offset, last_city_id, tail)


    @numba.jit('i8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:,:], f8)', nopython=True, parallel=False)
    def best_score_permutation_index(offset, head, firsts, lasts, lens, tail, scores, indexes, best_score):
        best_index = -1
        for i in numba.prange(len(indexes)):
            score = score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes[i])
            if score < best_score:
                best_index, best_score = i, score
        return best_index


    kdt = KDTree(XY)

    fives = set()
    closest = []
    for i in tqdm(cities.index):
        dists, neibs = kdt.query([XY[i]], 9)
        closest.append(dists[0, 1])
        for comb in combinations(neibs[0], 5):
            if all(comb):
                fives.add(tuple(sorted(comb)))
        neibs = kdt.query_radius([XY[i]], 10, count_only=False, return_distance=False)
        for comb in combinations(neibs[0], 5):
            if all(comb):
                fives.add(tuple(sorted(comb)))

    print(f'{len(fives)} cities fives are selected.')


    # sort fives by distance
    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def sum_distance(ids):
        res = 0
        for i in numba.prange(len(ids)):
            for j in numba.prange(i + 1, len(ids)):
                res += cities_distance(0, ids[i], ids[j])
        return res


    fives = np.array(list(fives))
    closest = np.array(closest)
    distances = np.array(list(map(sum_distance, tqdm(fives))))
    distances = distances / (closest[fives.T[0]] + closest[fives.T[1]] + closest[fives.T[2]] + closest[fives.T[3]] + closest[fives.T[4]])
    order = distances.argsort()
    fives = fives[order]
    # Sergi 2019-01-08 20:50:00 Lo cambio para ver si mejora
    #fives = fives[:2 * 10 ** 6]
    #fives = fives[2 * 10 ** 6:4 * 10 ** 6]
    #fives = fives[2 * 10 ** 6:5 * 10 ** 6]
    fives = fives[2 * 10 ** 6:8 * 10 ** 6]

    import gc
    gc.collect()

    path = pd.read_csv(filename).Path.values


    @lru_cache(maxsize=None)
    def indexes_permutations(n):
        return np.array(list(map(list, permutations(range(n)))))


    path_index = np.argsort(path[:-1])
    print(f'Total score is {score_path(path):.2f}.')
    for _ in range(2):
        for ids in tqdm(fives):
            i1, i2, i3, i4, i5 = np.sort(path_index[ids])
            head, tail = path[i1-1], path[i5+1]
            chunks = [path[i1:i1+1], path[i1+1:i2], path[i2:i2+1], path[i2+1:i3],
                      path[i3:i3+1], path[i3+1:i4], path[i4:i4+1], path[i4+1:i5], path[i5:i5+1]]
            chunks = [chunk for chunk in chunks if len(chunk)]
            scores = np.array([chunk_scores(chunk) for chunk in chunks])
            lens = np.array([len(chunk) for chunk in chunks])
            firsts = np.array([chunk[0] for chunk in chunks])
            lasts = np.array([chunk[-1] for chunk in chunks])
            best_score = score_compound_chunk(i1-1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks))[0])
            index = best_score_permutation_index(i1-1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks)), best_score)
            if index > 0:
                perm = [chunks[i] for i in indexes_permutations(len(chunks))[index]]
                path[i1-1:i5+2] = np.concatenate([[head], np.concatenate(perm), [tail]])
                path_index = np.argsort(path[:-1])
                print(f'New total score is {score_path(path):.3f}. Permutating path at indexes {i1}, {i2}, {i3}, {i4}, {i5}.')

    best_score = score_path(path)
    if path_out != '':
        assert (path_out != filename)
        pd.DataFrame({'Path': path}).to_csv(path_out + '_' + str(int(best_score)), index=False)

def numba_7comb(filename='1219_LKH_MT8_AWS/sub_2opt_1516029', path_out=''):

    cities = pd.read_csv('input/cities.csv', index_col=['CityId'])
    XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
    is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)

    @numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
    def cities_distance(offset, id_from, id_to):
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        if offset % 10 == 9 and is_not_prime[id_from]:
            return 1.1 * distance
        return distance


    @numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
    def score_chunk(offset, chunk):
        pure_distance, penalty = 0.0, 0.0
        penalty_modulo = 9 - offset % 10
        for path_index in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[path_index], chunk[path_index+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
                penalty += distance
        return pure_distance + 0.1 * penalty


    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def score_path(path):
        return score_chunk(0, path)


    @numba.jit
    def chunk_scores(chunk):
        scores = np.zeros(10)
        pure_distance = 0
        for i in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[i], chunk[i+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if is_not_prime[id_from]:
                scores[9-i%10] += distance
        scores *= 0.1
        scores += pure_distance
        return scores


    @numba.jit('f8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:])', nopython=True, parallel=False)
    def score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes):
        score = 0.0
        last_city_id = head
        for i in numba.prange(len(indexes)):
            index = indexes[i]
            first, last, chunk_len = firsts[index], lasts[index], lens[index]
            score += cities_distance(offset, last_city_id, first)
            score += scores[index, (offset + 1) % 10]
            last_city_id = last
            offset += chunk_len
        return score + cities_distance(offset, last_city_id, tail)


    @numba.jit('i8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:,:], f8)', nopython=True, parallel=False)
    def best_score_permutation_index(offset, head, firsts, lasts, lens, tail, scores, indexes, best_score):
        best_index = -1
        for i in numba.prange(len(indexes)):
            score = score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes[i])
            if score < best_score:
                best_index, best_score = i, score
        return best_index


    kdt = KDTree(XY)
    closest = []
    for i in tqdm(cities.index):
        dists, neibs = kdt.query([XY[i]], 10)
        closest.append(dists[0, 1])

    closest = np.array(closest)

    path_init = pd.read_csv(filename).Path.values
    path_index_init = np.argsort(path_init[:-1])

    path = pd.read_csv(filename).Path.values
    path_index = np.argsort(path[:-1])

    stages = 4

    for stage in range(stages):
        print('stage num: '+str(stage))
        path_index_init_0 = int(stage/stages*path_index_init.shape[0])
        path_index_init_1 = int((stage+1)/stages*path_index_init.shape[0])

        fives = set()

        for i in tqdm(path_index_init[path_index_init_0:path_index_init_1]):
            dists, neibs = kdt.query([XY[i]], 10)
            for comb in combinations(neibs[0], 7):
                if all(comb):
                    if max(path_index[list(comb)]) - min(path_index[list(comb)]) > 20:
                        fives.add(tuple(sorted(comb)))
            neibs = kdt.query_radius([XY[i]], 10, count_only=False, return_distance=False)
            for comb in combinations(neibs[0], 7):
                if all(comb):
                    if max(path_index[list(comb)]) - min(path_index[list(comb)]) > 20:
                        fives.add(tuple(sorted(comb)))

        print(f'{len(fives)} cities fives are selected.')


        # sort fives by distance
        @numba.jit('f8(i8[:])', nopython=True, parallel=False)
        def sum_distance(ids):
            res = 0
            for i in numba.prange(len(ids)):
                for j in numba.prange(i + 1, len(ids)):
                    res += cities_distance(0, ids[i], ids[j])
            return res


        fives = np.array(list(fives))

        distances = np.array(list(map(sum_distance, tqdm(fives))))
        distances = distances / (closest[fives.T[0]] + closest[fives.T[1]] + closest[fives.T[2]] + closest[fives.T[3]]
                                 + closest[fives.T[4]] + closest[fives.T[5]] + closest[fives.T[6]])
        order = distances.argsort()
        fives = fives[order]
        # fives = fives[:2 * 10 ** 6]

        import gc
        gc.collect()


        @lru_cache(maxsize=None)
        def indexes_permutations(n):
            return np.array(list(map(list, permutations(range(n)))))

        # @lru_cache(maxsize=None)
        # def indexes_permutations2(n):
        #     return np.array([p for p in list(map(list, permutations(range(n))))
        #                      if np.all(p == np.array([0,1,2,3,4,5,6])) or
        #                      ((p[1] - p[0] != 1) and (p[2] - p[1] != 1) and (p[3] - p[2] != 1) and (p[4] - p[3] != 1) and
        #                      (p[5] - p[4] != 1) and (p[6] - p[5] != 1) and (p[6] != 6) and (p[0] != 0))])


        path_index = np.argsort(path[:-1])
        print(f'Total score is {score_path(path):.2f}.')
        for _ in range(1):
            for ids in tqdm(fives):
                i1, i2, i3, i4, i5, i6, i7 = np.sort(path_index[ids])
                head, tail = path[i1-1], path[i7+1]
                chunks = [path[i1:i2], path[i2:i3], path[i3:i4], path[i4:i5], path[i5:i6], path[i6:i7], path[i7:i7 + 1]]
                # chunks = [chunk for chunk in chunks if len(chunk)]
                scores = np.array([chunk_scores(chunk) for chunk in chunks])
                lens = np.array([len(chunk) for chunk in chunks])
                firsts = np.array([chunk[0] for chunk in chunks])
                lasts = np.array([chunk[-1] for chunk in chunks])
                # best_score = score_compound_chunk(i1-1, head, firsts, lasts, lens, tail, scores, np.array([0,1,2,3,4,5,6]))
                best_score = score_compound_chunk(i1 - 1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks))[0])
                index = best_score_permutation_index(i1-1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks)), best_score)
                if index > 0:
                    perm = [chunks[i] for i in indexes_permutations(len(chunks))[index]]
                    path[i1-1:i7+2] = np.concatenate([[head], np.concatenate(perm), [tail]])
                    path_index = np.argsort(path[:-1])
                    print(f'New total score is {score_path(path):.3f}. Permutating path at indexes {i1}, {i2}, {i3}, {i4}, {i5}, {i6}, {i7}.')

        best_score = score_path(path)
        if path_out != '':
            assert (path_out != filename)
            pd.DataFrame({'Path': path}).to_csv(path_out + '_' + str(int(best_score)), index=False)

def numba_8comb(filename='1219_LKH_MT8_AWS/sub_2opt_1516029', path_out=''):

    cities = pd.read_csv('input/cities.csv', index_col=['CityId'])
    XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
    is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)

    @numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
    def cities_distance(offset, id_from, id_to):
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        if offset % 10 == 9 and is_not_prime[id_from]:
            return 1.1 * distance
        return distance


    @numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
    def score_chunk(offset, chunk):
        pure_distance, penalty = 0.0, 0.0
        penalty_modulo = 9 - offset % 10
        for path_index in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[path_index], chunk[path_index+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
                penalty += distance
        return pure_distance + 0.1 * penalty


    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def score_path(path):
        return score_chunk(0, path)


    @numba.jit
    def chunk_scores(chunk):
        scores = np.zeros(10)
        pure_distance = 0
        for i in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[i], chunk[i+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if is_not_prime[id_from]:
                scores[9-i%10] += distance
        scores *= 0.1
        scores += pure_distance
        return scores


    @numba.jit('f8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:])', nopython=True, parallel=False)
    def score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes):
        score = 0.0
        last_city_id = head
        for i in numba.prange(len(indexes)):
            index = indexes[i]
            first, last, chunk_len = firsts[index], lasts[index], lens[index]
            score += cities_distance(offset, last_city_id, first)
            score += scores[index, (offset + 1) % 10]
            last_city_id = last
            offset += chunk_len
        return score + cities_distance(offset, last_city_id, tail)


    @numba.jit('i8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:,:], f8)', nopython=True, parallel=False)
    def best_score_permutation_index(offset, head, firsts, lasts, lens, tail, scores, indexes, best_score):
        best_index = -1
        for i in numba.prange(len(indexes)):
            score = score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes[i])
            if score < best_score:
                best_index, best_score = i, score
        return best_index


    kdt = KDTree(XY)


    path_init = pd.read_csv(filename).Path.values
    path_index_init = np.argsort(path_init[:-1])

    path = pd.read_csv(filename).Path.values
    path_index = np.argsort(path[:-1])

    stages = 5

    for stage in range(stages):
        print('stage num: '+str(stage))
        path_index_init_0 = int(stage/stages*path_index_init.shape[0])
        path_index_init_1 = int((stage+1)/stages*path_index_init.shape[0])

        fives = set()
        for i in tqdm(path_index_init[path_index_init_0:path_index_init_1]):
            dists, neibs = kdt.query([XY[i]], 11)
            for comb in combinations(neibs[0], 8):
                if all(comb):
                    if max(path_index[list(comb)]) - min(path_index[list(comb)]) > 20:
                        fives.add(tuple(sorted(comb)))
            neibs = kdt.query_radius([XY[i]], 9, count_only=False, return_distance=False)
            for comb in combinations(neibs[0], 8):
                if all(comb):
                    if max(path_index[list(comb)]) - min(path_index[list(comb)]) > 20:
                        fives.add(tuple(sorted(comb)))

        print(f'{len(fives)} cities fives are selected.')


        # sort fives by distance
        @numba.jit('f8(i8[:])', nopython=True, parallel=False)
        def sum_distance(ids):
            res = 0
            for i in numba.prange(len(ids)):
                for j in numba.prange(i + 1, len(ids)):
                    res += cities_distance(0, ids[i], ids[j])
            return res


        fives = np.array(list(fives))
        distances = np.array(list(map(sum_distance, tqdm(fives))))
        order = distances.argsort()
        fives = fives[order]
        # fives = fives[:2 * 10 ** 6]

        import gc
        gc.collect()

        #
        # @lru_cache(maxsize=None)
        # def indexes_permutations(n):
        #     return np.array(list(map(list, permutations(range(n)))))

        @lru_cache(maxsize=None)
        def indexes_permutations(n):
            return np.vstack(
                [np.array([0,1,2,3,4,5,6,7]),
                 np.array([p for p in list(map(list, permutations(range(n))))
                           if ((p[1] - p[0] != 1) and (p[2] - p[1] != 1) and (p[3] - p[2] != 1) and (p[4] - p[3] != 1)
                               and (p[5] - p[4] != 1) and (p[6] - p[5] != 1) and (p[7] - p[6] != 1) and (p[7] != 7) and
                               (p[0] != 0))])
                 ])


        path_index = np.argsort(path[:-1])
        print(f'Total score is {score_path(path):.2f}.')
        for _ in range(1):
            for ids in tqdm(fives):
                i1, i2, i3, i4, i5, i6, i7, i8 = np.sort(path_index[ids])
                head, tail = path[i1-1], path[i8+1]
                chunks = [path[i1:i2], path[i2:i3], path[i3:i4], path[i4:i5], path[i5:i6], path[i6:i7],
                          path[i7:i8], path[i8:i8 + 1]]
                # chunks = [chunk for chunk in chunks if len(chunk)]
                scores = np.array([chunk_scores(chunk) for chunk in chunks])
                lens = np.array([len(chunk) for chunk in chunks])
                firsts = np.array([chunk[0] for chunk in chunks])
                lasts = np.array([chunk[-1] for chunk in chunks])
                best_score = score_compound_chunk(i1 - 1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks))[0])
                index = best_score_permutation_index(i1-1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks)), best_score)
                if index > 0:
                    perm = [chunks[i] for i in indexes_permutations(len(chunks))[index]]
                    path[i1-1:i8+2] = np.concatenate([[head], np.concatenate(perm), [tail]])
                    path_index = np.argsort(path[:-1])
                    print(f'New total score is {score_path(path):.3f}. Permutating path at indexes {i1}, {i2}, {i3}, {i4}, {i5}, {i6}, {i7}, {i8}.')

        best_score = score_path(path)
        if path_out != '':
            assert (path_out != filename)
            pd.DataFrame({'Path': path}).to_csv(path_out + '_' + str(int(best_score)), index=False)

def numba_8comb_v2(filename='1219_LKH_MT8_AWS/sub_2opt_1516029', path_out=''):

    cities = pd.read_csv('input/cities.csv', index_col=['CityId'])
    XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
    is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)

    @numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
    def cities_distance(offset, id_from, id_to):
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        if offset % 10 == 9 and is_not_prime[id_from]:
            return 1.1 * distance
        return distance


    @numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
    def score_chunk(offset, chunk):
        pure_distance, penalty = 0.0, 0.0
        penalty_modulo = 9 - offset % 10
        for path_index in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[path_index], chunk[path_index+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
                penalty += distance
        return pure_distance + 0.1 * penalty


    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def score_path(path):
        return score_chunk(0, path)


    @numba.jit
    def chunk_scores(chunk):
        scores = np.zeros(10)
        pure_distance = 0
        for i in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[i], chunk[i+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if is_not_prime[id_from]:
                scores[9-i%10] += distance
        scores *= 0.1
        scores += pure_distance
        return scores


    @numba.jit('f8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:])', nopython=True, parallel=False)
    def score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes):
        score = 0.0
        last_city_id = head
        for i in numba.prange(len(indexes)):
            index = indexes[i]
            first, last, chunk_len = firsts[index], lasts[index], lens[index]
            score += cities_distance(offset, last_city_id, first)
            score += scores[index, (offset + 1) % 10]
            last_city_id = last
            offset += chunk_len
        return score + cities_distance(offset, last_city_id, tail)


    @numba.jit('i8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:,:], f8)', nopython=True, parallel=False)
    def best_score_permutation_index(offset, head, firsts, lasts, lens, tail, scores, indexes, best_score):
        best_index = -1
        for i in numba.prange(len(indexes)):
            score = score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes[i])
            if score < best_score:
                best_index, best_score = i, score
        return best_index


    kdt = KDTree(XY)


    path_init = pd.read_csv(filename).Path.values
    path_index_init = np.argsort(path_init[:-1])

    path = pd.read_csv(filename).Path.values
    path_index = np.argsort(path[:-1])

    stages = 10

    for stage in range(stages):
        print('stage num: '+str(stage))
        path_index_init_0 = int(stage/stages*path_index_init.shape[0])
        path_index_init_1 = int((stage+1)/stages*path_index_init.shape[0])

        fives = set()
        for i in tqdm(path_index_init[path_index_init_0:path_index_init_1]):
            dists, neibs = kdt.query([XY[i]], 11)
            for comb in combinations(neibs[0], 8):
                if all(comb):
                    if max(path_index[list(comb)]) - min(path_index[list(comb)]) > 20:
                        fives.add(tuple(sorted(comb)))
            neibs = kdt.query_radius([XY[i]], 9, count_only=False, return_distance=False)
            for comb in combinations(neibs[0], 8):
                if all(comb):
                    if max(path_index[list(comb)]) - min(path_index[list(comb)]) > 20:
                        fives.add(tuple(sorted(comb)))

        print(f'{len(fives)} cities fives are selected.')


        # sort fives by distance
        @numba.jit('f8(i8[:])', nopython=True, parallel=False)
        def sum_distance(ids):
            res = 0
            for i in numba.prange(len(ids)):
                for j in numba.prange(i + 1, len(ids)):
                    res += cities_distance(0, ids[i], ids[j])
            return res


        fives = np.array(list(fives))
        distances = np.array(list(map(sum_distance, tqdm(fives))))
        order = distances.argsort()
        fives = fives[order]
        # fives = fives[:2 * 10 ** 6]

        import gc
        gc.collect()

        #
        # @lru_cache(maxsize=None)
        # def indexes_permutations(n):
        #     return np.array(list(map(list, permutations(range(n)))))

        # @lru_cache(maxsize=None)
        # def indexes_permutations(n):
        #     return np.vstack(
        #         [#np.array([0,1,2,3,4,5,6,7]),
        #          np.array([p for p in list(map(list, permutations(range(n))))
        #                    if ((p[1] - p[0] != 1) and (p[2] - p[1] != 1) and (p[3] - p[2] != 1) and (p[4] - p[3] != 1)
        #                        and (p[5] - p[4] != 1) and (p[6] - p[5] != 1) and (p[7] - p[6] != 1) and (p[7] != 7) and
        #                        (p[0] != 0))])
        #          ])
        @lru_cache(maxsize=None)
        def indexes_permutations(n):
            return np.array(list(map(list, permutations(range(n)))))

        path_index = np.argsort(path[:-1])
        print(f'Total score is {score_path(path):.2f}.')
        for _ in range(1):
            for ids in tqdm(fives):
                i1, i2, i3, i4, i5, i6, i7, i8 = np.sort(path_index[ids])
                head, tail = path[i1-1], path[i8+1]
                chunks = [path[i1:i2], path[i2:i3], path[i3:i4], path[i4:i5], path[i5:i6], path[i6:i7],
                          path[i7:i8], path[i8:i8 + 1]]
                # chunks = [chunk for chunk in chunks if len(chunk)]
                scores = np.array([chunk_scores(chunk) for chunk in chunks])
                lens = np.array([len(chunk) for chunk in chunks])
                firsts = np.array([chunk[0] for chunk in chunks])
                lasts = np.array([chunk[-1] for chunk in chunks])
                best_score = score_compound_chunk(i1 - 1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks))[0])
                index = best_score_permutation_index(i1-1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks)), best_score)
                if index > 0:
                    perm = [chunks[i] for i in indexes_permutations(len(chunks))[index]]
                    path[i1-1:i8+2] = np.concatenate([[head], np.concatenate(perm), [tail]])
                    path_index = np.argsort(path[:-1])
                    print(f'New total score is {score_path(path):.3f}. Permutating path at indexes {i1}, {i2}, {i3}, {i4}, {i5}, {i6}, {i7}, {i8}.')

        best_score = score_path(path)
        if path_out != '':
            assert (path_out != filename)
            pd.DataFrame({'Path': path}).to_csv(path_out + '_' + str(int(best_score)), index=False)

def numba_7comb_rev(filename='1219_LKH_MT8_AWS/sub_2opt_1516029', path_out=''):

    cities = pd.read_csv('input/cities.csv', index_col=['CityId'])
    XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
    is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)

    @numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
    def cities_distance(offset, id_from, id_to):
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        if offset % 10 == 9 and is_not_prime[id_from]:
            return 1.1 * distance
        return distance


    @numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
    def score_chunk(offset, chunk):
        pure_distance, penalty = 0.0, 0.0
        penalty_modulo = 9 - offset % 10
        for path_index in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[path_index], chunk[path_index+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
                penalty += distance
        return pure_distance + 0.1 * penalty


    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def score_path(path):
        return score_chunk(0, path)


    @numba.jit
    def chunk_scores(chunk):
        scores = np.zeros(10)
        pure_distance = 0
        for i in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[i], chunk[i+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if is_not_prime[id_from]:
                scores[9-i%10] += distance
        scores *= 0.1
        scores += pure_distance
        return scores


    @numba.jit('f8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:])', nopython=True, parallel=False)
    def score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes):
        score = 0.0
        last_city_id = head
        for i in numba.prange(len(indexes)):
            index = indexes[i]
            first, last, chunk_len = firsts[index], lasts[index], lens[index]
            score += cities_distance(offset, last_city_id, first)
            score += scores[index, (offset + 1) % 10]
            last_city_id = last
            offset += chunk_len
        return score + cities_distance(offset, last_city_id, tail)


    @numba.jit('i8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:,:], f8)', nopython=True, parallel=False)
    def best_score_permutation_index(offset, head, firsts, lasts, lens, tail, scores, indexes, best_score):
        best_index = -1
        for i in numba.prange(len(indexes)):
            score = score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes[i])
            if score < best_score:
                best_index, best_score = i, score
        return best_index


    kdt = KDTree(XY)


    path_init = pd.read_csv(filename).Path.values
    path_index_init = np.argsort(path_init[:-1])

    path = pd.read_csv(filename).Path.values
    path_index = np.argsort(path[:-1])

    stages = 20

    for stage in range(stages):
        print('stage num: '+str(stage))
        path_index_init_0 = int(stage/stages*path_index_init.shape[0])
        path_index_init_1 = int((stage+1)/stages*path_index_init.shape[0])

        fives = set()
        for i in tqdm(path_index_init[path_index_init_0:path_index_init_1]):
            dists, neibs = kdt.query([XY[i]], 10)
            for comb in combinations(neibs[0], 7):
                if all(comb):
                    if max(path_index[list(comb)]) - min(path_index[list(comb)]) > 20:
                        fives.add(tuple(sorted(comb)))
            neibs = kdt.query_radius([XY[i]], 10, count_only=False, return_distance=False)
            for comb in combinations(neibs[0], 7):
                if all(comb):
                    if max(path_index[list(comb)]) - min(path_index[list(comb)]) > 20:
                        fives.add(tuple(sorted(comb)))

        print(f'{len(fives)} cities fives are selected.')


        # sort fives by distance
        @numba.jit('f8(i8[:])', nopython=True, parallel=False)
        def sum_distance(ids):
            res = 0
            for i in numba.prange(len(ids)):
                for j in numba.prange(i + 1, len(ids)):
                    res += cities_distance(0, ids[i], ids[j])
            return res


        fives = np.array(list(fives))
        distances = np.array(list(map(sum_distance, tqdm(fives))))
        order = distances.argsort()
        fives = fives[order]
        # fives = fives[:2 * 10 ** 6]

        import gc
        gc.collect()


        @lru_cache(maxsize=None)
        def indexes_permutations(n):
            return np.array(list(map(list, permutations(range(n)))))

        # @lru_cache(maxsize=None)
        # def indexes_permutations2(n):
        #     return np.array([p for p in list(map(list, permutations(range(n))))
        #                      if np.all(p == np.array([0,1,2,3,4,5,6])) or
        #                      ((p[1] - p[0] != 1) and (p[2] - p[1] != 1) and (p[3] - p[2] != 1) and (p[4] - p[3] != 1) and
        #                      (p[5] - p[4] != 1) and (p[6] - p[5] != 1) and (p[6] != 6) and (p[0] != 0))])


        path_index = np.argsort(path[:-1])
        print(f'Total score is {score_path(path):.2f}.')
        for _ in range(1):
            for ids in tqdm(fives):
                i1, i2, i3, i4, i5, i6, i7 = np.sort(path_index[ids])
                head, tail = path[i1-1], path[i7+1]
                chunks = [path[i1:i2], path[i2:i3], path[i3:i4], path[i4:i5], path[i5:i6], path[i6:i7], path[i7:i7 + 1]]
                # chunks = [chunk for chunk in chunks if len(chunk)]
                scores = np.array([chunk_scores(chunk) for chunk in chunks])
                lens = np.array([len(chunk) for chunk in chunks])
                lens_max_pos = lens.argmax()
                firsts = np.array([chunk[0] for chunk in chunks])
                lasts = np.array([chunk[-1] for chunk in chunks])
                best_score = score_compound_chunk(i1 - 1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks))[0])
                index = best_score_permutation_index(i1-1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks)), best_score)
                if index > 0:
                    perm = [chunks[i] for i in indexes_permutations(len(chunks))[index]]
                    path[i1-1:i7+2] = np.concatenate([[head], np.concatenate(perm), [tail]])
                    path_index = np.argsort(path[:-1])
                    print(f'New total score is {score_path(path):.3f}. Permutating path at indexes {i1}, {i2}, {i3}, {i4}, {i5}, {i6}, {i7}.')
                else:
                    chunks[lens_max_pos] = chunks[lens_max_pos][::-1]
                    index = best_score_permutation_index(i1 - 1, head, firsts, lasts, lens, tail, scores,
                                                         indexes_permutations(len(chunks)), best_score)
                    if index > 0:
                        perm = [chunks[i] for i in indexes_permutations(len(chunks))[index]]
                        path[i1 - 1:i7 + 2] = np.concatenate([[head], np.concatenate(perm), [tail]])
                        path_index = np.argsort(path[:-1])
                        print(f'New total score is {score_path(path):.3f}. Permutating path at indexes {i1}, {i2}, {i3}, {i4}, {i5}, {i6}, {i7}.')
        best_score = score_path(path)
        if path_out != '':
            assert (path_out != filename)
            pd.DataFrame({'Path': path}).to_csv(path_out + '_' + str(int(best_score)), index=False)

def numba_3comb_large(filename='1219_LKH_MT8_AWS/sub_2opt_1516029', path_out=''):

    cities = pd.read_csv('input/cities.csv', index_col=['CityId'])
    XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
    is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)

    @numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
    def cities_distance(offset, id_from, id_to):
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        if offset % 10 == 9 and is_not_prime[id_from]:
            return 1.1 * distance
        return distance


    @numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
    def score_chunk(offset, chunk):
        pure_distance, penalty = 0.0, 0.0
        penalty_modulo = 9 - offset % 10
        for path_index in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[path_index], chunk[path_index+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
                penalty += distance
        return pure_distance + 0.1 * penalty


    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def score_path(path):
        return score_chunk(0, path)


    @numba.jit
    def chunk_scores(chunk):
        scores = np.zeros(10)
        pure_distance = 0
        for i in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[i], chunk[i+1]
            xy_from, xy_to = XY[id_from], XY[id_to]
            dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
            distance = sqrt(dx * dx + dy * dy)
            pure_distance += distance
            if is_not_prime[id_from]:
                scores[9-i%10] += distance
        scores *= 0.1
        scores += pure_distance
        return scores


    @numba.jit('f8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:])', nopython=True, parallel=False)
    def score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes):
        score = 0.0
        last_city_id = head
        for i in numba.prange(len(indexes)):
            index = indexes[i]
            first, last, chunk_len = firsts[index], lasts[index], lens[index]
            score += cities_distance(offset, last_city_id, first)
            score += scores[index, (offset + 1) % 10]
            last_city_id = last
            offset += chunk_len
        return score + cities_distance(offset, last_city_id, tail)


    @numba.jit('i8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:,:], f8)', nopython=True, parallel=False)
    def best_score_permutation_index(offset, head, firsts, lasts, lens, tail, scores, indexes, best_score):
        best_index = -1
        for i in numba.prange(len(indexes)):
            score = score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes[i])
            if score < best_score:
                best_index, best_score = i, score
        return best_index


    kdt = KDTree(XY)

    fives = set()
    for i in tqdm(cities.index):
        dists, neibs = kdt.query([XY[i]], 14)
        if dists[0,3]>12:
            for comb in combinations(neibs[0], 4):
                if all(comb):
                    fives.add(tuple(sorted(comb)))

    print(f'{len(fives)} cities fives are selected.')


    # sort fives by distance
    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def sum_distance(ids):
        res = 0
        for i in numba.prange(len(ids)):
            for j in numba.prange(i + 1, len(ids)):
                res += cities_distance(0, ids[i], ids[j])
        return res


    fives = np.array(list(fives))
    distances = np.array(list(map(sum_distance, tqdm(fives))))
    order = distances.argsort()
    fives = fives[order]
    # fives = fives[:2 * 10 ** 6]

    import gc
    gc.collect()

    path = pd.read_csv(filename).Path.values


    @lru_cache(maxsize=None)
    def indexes_permutations(n):
        return np.array(list(map(list, permutations(range(n)))))


    path_index = np.argsort(path[:-1])
    print(f'Total score is {score_path(path):.2f}.')
    for _ in range(1):
        for ids in tqdm(fives):
            i1, i2, i3, i4 = np.sort(path_index[ids])
            head, tail = path[i1-1], path[i4+1]
            chunks = [path[i1:i1+1], path[i1+1:i2], path[i2:i2+1], path[i2+1:i3],
                      path[i3:i3+1], path[i3+1:i4], path[i4:i4+1]]
            chunks = [chunk for chunk in chunks if len(chunk)]
            scores = np.array([chunk_scores(chunk) for chunk in chunks])
            lens = np.array([len(chunk) for chunk in chunks])
            firsts = np.array([chunk[0] for chunk in chunks])
            lasts = np.array([chunk[-1] for chunk in chunks])
            best_score = score_compound_chunk(i1-1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks))[0])
            index = best_score_permutation_index(i1-1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks)), best_score)
            if index > 0:
                perm = [chunks[i] for i in indexes_permutations(len(chunks))[index]]
                path[i1-1:i4+2] = np.concatenate([[head], np.concatenate(perm), [tail]])
                path_index = np.argsort(path[:-1])
                print(f'New total score is {score_path(path):.3f}. Permutating path at indexes {i1}, {i2}, {i3}, {i4}.')

    best_score = score_path(path)
    if path_out != '':
        assert (path_out != filename)
        pd.DataFrame({'Path': path}).to_csv(path_out + '_' + str(int(best_score)), index=False)

# LKH to mini paths
def subtram_lkh_rec2(file_path='1214_LKH_MC2_local/sub_2opt_1516258', path_out='', max_i=500, tempdir='temp2',
                    sub_size = 100, loops = 5, lkh_path = '/home/luis/src'):
    # final_results = []pen_list
    x = time.time()
    cities = pd.read_csv('./input/cities.csv')
    np_xy_cities = cities[['X', 'Y']].values

    sub = pd.read_csv(file_path).Path.values

    old_score = calc_score(sub, cities)
    print('old score: ' + str(old_score))
    dists, pens = calc_2_vec_score(sub, cities)

    for i in tqdm(range(max_i)):
        max_pos = int((sub.shape[0] - sub_size)*i/max_i)

        subtour = sub[max_pos:(max_pos + sub_size)]
        sub_c = sub.copy()

        old_score_2 = calc_score(sub, cities)
        best_score = old_score_2

        subpens = pens[max_pos:(max_pos + sub_size - 1)]
        subpens_pos = np.argsort(subpens)[::-1]

        # print('mean dev of new path: ' + str(np.std(subtour[1:] - subtour[:-1])) + ' ... ' + str(old_score_2))
        # print(pen_matrix[85, 86])
        for loop in range(loops):

            pen_matrix = np.zeros([sub_size, sub_size])
            # for j in range(sub_size-1):
            j = subpens_pos[loop]
            pen_matrix[j, j + 1] = subpens[j]

            parameters = [
                ("PROBLEM_FILE", tempdir + "/mat_.tsp"),
                ("OUTPUT_TOUR_FILE", tempdir + "/tsp_solution.csv"),
                ("SEED", 2018),
                ('CANDIDATE_SET_TYPE', 'POPMUSIC'),  # 'NEAREST-NEIGHBOR', 'ALPHA'),
            ]
            write_parameters_temp(parameters, filename=tempdir + '/mat_.par')
            write_tsp_file_temp2(subtour, np_xy_cities, filename=tempdir + '/mat_.tsp', name='ts', pen_matrix=pen_matrix)

            os.system(lkh_path + '/LKH-2.0.9/LKH ' + tempdir + '/mat_.par > /dev/null')

            path_ = np_read_output_LKH(tempdir + '/tsp_solution.csv')[:-1]

            path = subtour[path_]
            sub_c[max_pos:(max_pos + sub_size)] = path
            score_loop = calc_score(sub_c, cities)
            # print('mean dev of new path: ' + str(np.std(path[1:] - path[:-1])) + ' ... ' + str(score_loop))

            # dists_, pens_ = calc_2_vec_score(sub_c, cities)
            # subpens_ = pens_[max_pos:(max_pos + sub_size - 1)]

            # pen_matrix_new = np.zeros([sub_size, sub_size])
            # for j in range(max_i - 1):
            #     if subpens_[j] > 0:
            #         max_j = max(path_[j], path_[j + 1])
            #         min_j = min(path_[j], path_[j + 1])
            #         pen_matrix_new[min_j, max_j] = subpens_[j]

            # pen_matrix = (pen_matrix * (loop + 1) / (loop + 2) + pen_matrix_new * (1 / (loop + 2)))*multiply
            # print(pen_matrix[85, 86])
            if score_loop < best_score:
                best_score = score_loop
                best_sub = sub_c.copy()
                # best_multiplier = multiplier

        if best_score < old_score_2:
            print("saving: " + str(old_score_2 - best_score) + ' ... ' + 'current_score: ' + str(int(best_score)))
            sub = best_sub.copy()

    final_score = calc_score(sub, cities)
    print("saving: " + str(old_score - best_score))
    print("final score: " + str(final_score))
    print(str((time.time() - x) / 60) + ' minutes')

    # final_results = pd.concat(final_results)
    # final_results['under0'] = final_results['diff_score'] < 0
    # under0_mean = final_results.groupby('multiplier')['under0'].mean()
    # under0_exec = final_results.groupby('try')['under0'].mean()
    # print(final_results.groupby('try')['diff_score'].min().mean() / subtour.shape[0] * sub.shape[0])

    pd.DataFrame({'Path': sub}).to_csv(path_out + '501opt_' + str(int(final_score)), index=False)

def pseudo_k_opt(file_path='1214_LKH_MC2_local/sub_2opt_1516258', path_out='', k_opt = 4, num_neighbors = 5, radius = 0):

    K = k_opt
    NUM_NEIGHBORS = num_neighbors
    RADIUS = radius
    print("\n K: ", K)
    print("\n NUM_NEIGHBORS: ", NUM_NEIGHBORS)
    print("\n RADIUS: ", RADIUS)

    x = time.time()
    cities = pd.read_csv('./input/cities.csv', index_col = ['CityId'])
    np_xy_cities = cities[['X', 'Y']].values
    XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
    is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)

    @numba.jit('f8(i8, i8)', nopython=True, parallel=False)
    def euclidean_distance(id_from, id_to):
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        return sqrt(dx * dx + dy * dy)


    @numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
    def cities_distance(offset, id_from, id_to):
        """Euclidean distance with prime penalty."""
        distance = euclidean_distance(id_from, id_to)
        if offset % 10 == 9 and is_not_prime[id_from]:
            return 1.1 * distance
        return distance


    @numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
    def score_chunk(offset, chunk):
        """Return the total score (distance) for a chunk (array of cities)."""
        pure_distance, penalty = 0.0, 0.0
        penalty_modulo = 9 - offset % 10
        for path_index in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[path_index], chunk[path_index+1]
            distance = euclidean_distance(id_from, id_to)
            pure_distance += distance
            if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
                penalty += distance
        return pure_distance + 0.1 * penalty


    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def score_path(path):
        return score_chunk(0, path)

    @numba.jit
    def chunk_scores(chunk):
        scores = np.zeros(10)
        pure_distance = 0
        for i in numba.prange(chunk.shape[0] - 1):
            id_from, id_to = chunk[i], chunk[i+1]
            distance = euclidean_distance(id_from, id_to)
            pure_distance += distance
            if is_not_prime[id_from]:
                scores[9-i%10] += distance
        scores *= 0.1
        scores += pure_distance
        return scores

    def score_compound_chunk(offset, head, chunks, tail, scores, indexes_permutation=None):
        """
        Return the total distance for the path formed by all chunks in the
        order defined by the last argument.
        """
        if indexes_permutation is None:
            indexes_permutation = range(len(chunks))
        score = 0.0
        last_city_id = head
        for index in indexes_permutation:
            chunk, chunk_scores = chunks[index], scores[index]
            score += cities_distance(offset % 10, last_city_id, chunk[0])
            score += chunk_scores[(offset + 1) % 10]
            last_city_id = chunk[-1]
            offset += len(chunk)
        return score + cities_distance(offset % 10, last_city_id, tail)

    # sort candidates by distance
    @numba.jit('f8(i8[:])', nopython=True, parallel=False)
    def sum_distance(ids):
        res = 0
        for i in numba.prange(len(ids)):
            for j in numba.prange(i + 1, len(ids)):
                res += cities_distance(0, ids[i], ids[j])
        return res

    def not_trivial_permutations(iterable):
        perms = permutations(iterable)
        next(perms)
        yield from perms

    @lru_cache(maxsize=None)
    def not_trivial_indexes_permutations(length):
        return np.array([list(p) for p in not_trivial_permutations(range(length))])

    sub = pd.read_csv(file_path).Path.values
    path = pd.read_csv(file_path, squeeze=True).values

    initial_score = calc_score(sub, cities)
    print("Initial tour distance (score): {:.2f}".format(initial_score))

    kdt = KDTree(XY)
    candidates = set()
    for city_id in tqdm(cities.index):
        # Find N nearest neighbors
        dists, neibs = kdt.query([XY[city_id]], NUM_NEIGHBORS)
        for candidate in combinations(neibs[0], K):
            if all(candidate):
                candidates.add(tuple(sorted(candidate)))
        # Also add all cities in a given range (radius)
        neibs = kdt.query_radius([XY[city_id]], RADIUS, count_only=False, return_distance=False)
        for candidate in combinations(neibs[0], K):
            if all(candidate):
                candidates.add(tuple(sorted(candidate)))

    print("{} groups of {} cities are selected.".format(len(candidates), K))

    candidates = np.array(list(candidates))
    distances = np.array(list(map(sum_distance, tqdm(candidates))))
    order = distances.argsort()
    candidates = candidates[order]

    path_index = np.argsort(sub[:-1])
    for _ in range(1):
        for ids in tqdm(candidates):
            #ids = list(ids)
            #print(type(ids))
            #print(type(path_index))
            #print(type(sub))
            #print(path_index.shape)
            #print(sub.shape)
            # Index for each city in the order they appear in tour
            idx = sorted(path_index[ids])
            head, tail = path[idx[0] - 1], path[idx[-1] + 1]
            # Split the path between the candidate cities
            chunks = [path[idx[0]:idx[0]+1]]
            for i in range(len(idx) - 1):
                chunks.append(path[idx[i]+1:idx[i+1]])
                chunks.append(path[idx[i+1]:idx[i+1]+1])
            # Remove empty chunks and calculate score for each remaining chunk
            chunks = [chunk for chunk in chunks if len(chunk)]
            scores = [chunk_scores(chunk) for chunk in chunks]
            # Distance (score) for all chunks in the current order
            default_score = score_compound_chunk(idx[0]-1, head, chunks, tail, scores)
            best_score = default_score
            for indexes_permutation in not_trivial_indexes_permutations(len(chunks)):
                # Get score for all chunks when permutating the order
                score = score_compound_chunk(idx[0]-1, head, chunks, tail, scores, indexes_permutation)
                if score < best_score:
                    permutation = [chunks[i] for i in indexes_permutation]
                    best_chunk = np.concatenate([[head], np.concatenate(permutation), [tail]])
                    best_score = score
                if best_score < default_score:
                    # Update tour if an improvement has been found
                    path[idx[0]-1:idx[-1]+2] = best_chunk
                    path_index = np.argsort(path[:-1])
                    improvement = True

    final_score = score_path(path)
    print("Final score is {:.2f}, improvement: {:.2f}".format(final_score, initial_score - final_score))
    name_file = path_out + str(int(final_score)) + '.csv'
    pd.DataFrame({'Path': path}).to_csv(name_file, index=False)