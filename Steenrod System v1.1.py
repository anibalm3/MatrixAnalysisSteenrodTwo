# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 20:58:08 2018

@author: amedinam
"""

from itertools import product
'''we work with frozensets to represent simplices and sets of them to
represent mod 2 linear combinations'''


def partial(linear_combination):
    '''computes the boundary of a linear combination in C or CxC''' 
    def basis_partial(simplex):
        basis_boundary = set()
        for vertex in simplex:
            di_simplex = set(simplex) #clone, not reference
            di_simplex.discard(vertex)
            basis_boundary.add(frozenset(di_simplex))
        return basis_boundary

    boundary = set()
    if isinstance(linear_combination, frozenset):
        return basis_partial(linear_combination) 
    
    elif all(isinstance(x, frozenset) for x in linear_combination): 
        for term in linear_combination:
            basis_boundary = basis_partial(term)
            boundary.symmetric_difference_update(basis_boundary)

    elif all(isinstance(x[0], tuple) and len(x)==2 for x in linear_combination):
        for term in linear_combination:
            partial_first_factor = basis_partial(term[0])
            partial_second_factor = basis_partial(term[1])
            s1 = set(product(partial_first_factor,{term[1]}))
            s2 = set(product({term[0]},partial_second_factor)) 
            boundary.symmetric_difference_update(s1.symmetric_difference(s2))
    else:
        print('wrong type of input')
        return

    boundary.discard(())
    return boundary

print(partial({1}))