import itertools


def getset(daset, bigdat, min_s): # count frequency
    dic = {}
    for i in daset:
        dic[tuple(i)] = 0
    for i in bigdat:
        for j in daset:
            t = set(j)
            if t <= set(i):
                dic[tuple(j)] += 1
    dic = del_set(dic, min_s)
    print(dic)
    return sorted(dic.keys())


def del_set(dat, min_s): # delete items which is smaller than min support
    mark = []
    for k, v in dat.items():
        if v < min_s:
            mark.append(k)
    for i in mark:
        del dat[i]
    return dat


def apriori_gen(dat):  # generate candidate dataset
    nel = []
    l = len(dat)
    r = len(dat[0])
    for i in range(0, l):
        for j in range(i+1, l):
            if dat[i][0:r-1] == dat[j][0:r-1]:
                c = list(dat[i])
                c.append(dat[j][-1])
                if has_frequent_subset(c, dat):
                    nel.append(c)
    return nel


def has_frequent_subset(new, old): # judge if subset belong to old dataset
    return ~(False in [i in old for i in itertools.combinations(new, len(new)-1)])


def generate_llist(dat):
    a = []
    for i in dat:
        a.extend(i)
    li = set(a)
    flist = []
    for i in li:  # convert set to list
        t = []
        t.append(i)
        flist.append(t)
    return flist


def main(dat, min_s):
    stt = generate_llist(dat)
    print(stt)
    print(1, '  items set :')
    stt = getset(stt, dat, min_s)
    all = stt;
    i = 2
    while stt != []:
        print(i, '  items set :')
        i += 1
        al = apriori_gen(stt)
        if al == []:
            break
        stt = getset(al, dat, min_s)
        all.append(stt)
    print(all)


data = [['A', 'B', 'C', 'D'],
        ['B', 'C', 'E'],
        ['A', 'B', 'C', 'E'],
        ['B', 'D', 'E'],
        ['A', 'B', 'C', 'D']]
min_support = len(data) * 0.4
main(data, min_support)
# data = []
# for line in open('groceries.csv'):
#     data.append(line[:-1].split(","))
# min_support = len(data) * 0.001
# print(min_support)
# main(data, min_support)
