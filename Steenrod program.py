from random import randint

def tuples_matrix(d, n, constraints):
    '''Returns the matrix with d rows and n columns,
    where cell [n - 1][d - 1] contains the n-tuples that sum
    up to d - 1 and contain n elements. n should always be > than d'''

    result = []
    for i in range(d):
        result.append([{(i,)}])
    print(result)
    result[0] = [{(0,) * i} for i in range(1, n + 1)]
    
    
    col = 1
    for p, c in constraints:
        while col <= p:
            for i in range(1, d):
                cell = set()
                for j in range(i + 1):
                    for t2 in result[i - j][col - 1]:
                        cell.add((j,) + t2)
                result[i].append(cell)

            col += 1

        for i in range(d):
            to_remove = set()
            for tupla in result[i][p]:
                if sum(tupla[0:c]) == 0:
                    to_remove.add(tupla)
            result[i][p] -= to_remove

    while col < n:
        for i in range(1, d):
            cell = set()
            for j in range(i + 1):
                for t2 in result[i - j][col - 1]:
                    cell.add((j,) + t2)
            result[i].append(cell)

        col += 1

    return result


def representative_to_constraints(rep):
    '''given a representative in the form of a tuple, returns a
    list of pairs (start, length) representing the constraints imposed'''
    const = set()
    for i in range(len(rep)):
        n_list = rep[i + 1:]
        if rep[i] in n_list:
            const.add((i + 1, n_list.index(rep[i])))
    return const


# @TODO: minimize set of constraints, improves efficiency
def min_constraints(constraints):
    pass


def stringify(tupla, representative):
    '''given a tuple and a representative, returns the string representing
    the 'level changes' in a human readable form'''

    d = max(representative) + 1  # number of levels
    strings = []  # matrix of chars

    for i in range(d):
        strings.append([" "] * (d + 1))

    counter = 0
    step = 0
    for jump in tupla:
        # write the numbers in the right level
        for i in range(jump + 1):
            strings[representative[step]][counter] = str(counter)
            counter += 1
        counter -= 1

        # make the level "jump", writing the counter at each level
        if step < len(representative) - 1:
            sign = 1 if representative[step] < representative[step + 1] else -1
            for i in range(representative[step], representative[step + 1], sign):
                strings[i][counter] = str(counter)
        step += 1

    return "\n".join(["".join(s) for s in strings])


if __name__ == '__main__':

    representative = (1,2,3,4,2,4)
    d = 6
    n = len(representative)
    constraints = representative_to_constraints(representative)
    matrix = tuples_matrix(d, n, constraints)
    
    print("rows: d=%d\ncols: n=%d" % (d, n))
    print("representative: %s" % str(representative))
    print("constraints: %s\n" % str(constraints))
    print("%d tuples found\n" % len(matrix[d - 1][n - 1]))
    
    # discard random amount of tuples
    for i in range(randint(0, min(30, len(matrix[d - 1][n - 1])))):
        matrix[d - 1][n - 1].pop()

    sample = matrix[d - 1][n - 1].pop()
    print("Example:\n")
    print("%s\n%s\n" % (str(representative), str(sample)))
    print(stringify(sample, representative))
