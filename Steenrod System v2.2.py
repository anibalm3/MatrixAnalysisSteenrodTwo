# -*- coding: utf-8 -*-
"""
Created on Sun May  6 12:01:47 2018

@author: Anibal
"""

import numpy as np
import itertools
from sympy import Matrix

dim = 2

def get_gr_gen(dim):
    '''returns the list whose d-entry is the lexicographically ordered list of 
    d-faces of the (dim)-simplex'''  

    gr_gen = []
    for i in range(dim+1):
        els = [list(x) for x in itertools.combinations(set(range(dim+1)), i+1)]
        gr_gen.append(els)
    
    return gr_gen

#print('\ngraded generators for dim =',dim)
#for d in range(dim+1):
#    print('in degree',d,'\n',get_gr_gen(dim)[d])
    

def get_gr_partial(dim): 
    '''returns the list whose d-entry is the np.array (matrix) representing the 
    boundary map from d to (d-1)-chains of the standard (dim)-simplex with bases 
    lexicographically ordered'''

    gr_gen = get_gr_gen(dim)         
    gr_partial = list()
    for degree in range(dim+1):
        
        col_len = len(gr_gen[degree])
        row_len = len(gr_gen[degree-1])
        partial_sub_degree = np.zeros((row_len,col_len), dtype=int) #initializing matrix
        
        for i in range(col_len): #for all generators in the given degree
            for vertex in range(len(gr_gen[degree][i])): #for all the vertices of a generator
                els = list(gr_gen[degree][i])
                del els[vertex]
                for j in range(row_len): #looking among all generators of lower degree
                    if els == gr_gen[degree-1][j]:
                        partial_sub_degree[j,i]=(-1)**int(vertex)
                        break
        
        gr_partial.append(partial_sub_degree)
    
    return(gr_partial)
    
#print('\ngraded partial for dim =',dim)
#for d in range(dim+1):
#    print('and degree',d,'\n',get_gr_partial(dim)[d])


def get_tensor_gr_gen(dim):        
    '''returns the list whose d-entry is the lexicographically ordered list of pairs of faces 
    of the (dim)-simplex whose dimensions add to d'''      

    gr_gen = get_gr_gen(dim)
    tensor_gr_gen = []    
    for i in range (2*dim+1): #initializing the array
        tensor_gr_gen.append([])            
    for i in range(dim+1): #filling the array
        for j in range(dim+1):
            els = [list(x) for x in itertools.product(gr_gen[i],gr_gen[j])]
            tensor_gr_gen[i+j] += els    
    return tensor_gr_gen

#print('\ngenerators for dim',dim)
#for d in range(2*dim+1):
#    print('in degree',d,'\n',get_tensor_gr_gen(dim)[d])
    

def get_tensor_partial(dim):
    '''computes the matrix representing the dim-boundary in the tensor product 
    (with respect to the canonical basis lexicografically ordered)'''

    gr_gen = get_gr_gen(dim)
    gr_dim = []
    for l in gr_gen:
        gr_dim.append(len(l))
    
    gr_partial = get_gr_partial(dim)
    
    #constructs the first column
    hor_stack = np.array([0]) 
    for i in range(dim):
        ver_len = gr_dim[i]*gr_dim[dim-1-i]
        block = np.full((ver_len,1),ver_len,dtype=int)
        hor_stack = np.vstack((hor_stack,block))
         
    #constructs and stacks the blocks
    for i in range(dim+1):        
        hor_len = gr_dim[i]*gr_dim[dim-i]
        ver_stack = np.full((1,hor_len),hor_len,dtype=int) #constructs the first row   
        ith_sign = 1
        
        for j in range(dim):
            ver_len = gr_dim[j]*gr_dim[dim-1-j]
    
            if j+1 == i: #dx1:
                identity = np.eye(gr_dim[dim-i],dtype=int)
                block = np.kron(gr_partial[i],identity)
                
            elif dim-j == dim-i: #1xd
                identity = np.eye(gr_dim[i],dtype=int)
                block = np.kron(identity,ith_sign*gr_partial[dim-i])
                
            else: 
                block = np.zeros((ver_len,hor_len),dtype=int)
            
            ver_stack = np.vstack((ver_stack,block))
            ith_sign *= -1
            
        hor_stack = np.hstack((hor_stack,ver_stack))
    
    #deleting first row and column
    hor_stack = np.delete(hor_stack, (0), axis=0)    
    out_matrix = np.delete(hor_stack, (0), axis=1)    

    return out_matrix   

#print('\npartial in degree',dim,'\n',get_tensor_partial(dim))
        

def get_diag_of_boundary(dim):
    '''returns the vector, with entries: -1, 0 or 1, representing the linear 
    combination obtained by taking boundary of the top dimensional generator 
    and then applying the standard diagonal'''
    
    
    def get_boundary(face):
        '''given a face it returns the lexicographically 
        order list of signed generators appearing in its boundary'''
    
        boundary = []
        ith_sign = 1
        for i in face:
            di_face = list(face)
            di_face.pop(i)
            boundary.append([ith_sign,di_face])
            ith_sign *= (-1)
        
        return(boundary)

    
    def get_diag(sign,face):
        '''given a signed face, [plus/minus 1, [u,v,...,w]], it returns the 
        list of signed generators, [plus/minus 1, [[u,v],[v,...,w]]], appearing 
        in its standard diagonal'''
        
        diag = []
        for i in face:
            diag.append([sign,face[:face.index(i)+1],face[face.index(i):]])
    
        return(diag)      
    
    
    simplex = list(range(dim+1))    
    boundary = get_boundary(simplex)
    
    diag_partial = []
    for face in boundary: #computes all signed terms in the diagonal of the boundary
        aux = get_diag(face[0],face[1])
        diag_partial.extend(aux)
        
    tensor_gen = get_tensor_gr_gen(dim)[dim-1]    
    out_vector = np.zeros((len(tensor_gen),1),dtype=int)    
    
    for gen in tensor_gen: #constructs the vector representing the linear combination
        for term in diag_partial:
            if gen == term[1:]:
                out_vector[tensor_gen.index(gen)] = term[0]
    
    return out_vector

#print('\ndiagonal of the boundary of the top generator\n',get_diag_of_boundary(dim))
    

def get_num_of_free_vars_and_relations(dim):
    '''row echelon reduces the augmented matrix representing the equation partial(Delta(dim))=(Delta(dim-1))partial 
    identifying the positions of the free columns and the relations determining the pivot columns w/r to these'''
    
    #getting the augmented matrix
    M = Matrix(get_tensor_partial(dim))
    len_col = M.shape[1]
    M = M.col_insert(len_col, Matrix(get_diag_of_boundary(dim)))
    
    #getting the row reduced matrix and the list of pivots
    red_matrix = M.rref()[0]
    pivots = M.rref()[1]
    
    #a dictionary mapping from the column number of a free variables to their position in the list of free variables
    dict_free_vars = {}
    j = 0
    for i in range(len_col+1):
        if i not in pivots:
            dict_free_vars[i] = j
            j += 1
            
    relations = []
    row = 0
    for i in range(len_col): #not including last column
        
        aux = []        
        if i not in pivots:
            aux = [dict_free_vars[i]]
            row += 1
            
        elif i in pivots:
            j = i+1
            while i < j < len_col+1:
                if red_matrix[i-row,j] != 0:
                        aux.append(dict_free_vars[j])
                j += 1    
                
        relations.append(aux)
    
    return [int(len(dict_free_vars)-1),relations]

#print('number of free variables\n',get_num_of_free_vars_and_relations(dim)[0])
#print('all relations\n',get_num_of_free_vars_and_relations(dim)[1])


def free_vars_to_sol_vect(free_vars,relations):
    '''from a choice of values for the free variables, it produces the associated solution vector'''
    
    sol_vect = []
    for i in range(len(relations)):
        aux = 1
        for j in relations[i]:
            aux *= free_vars[j]
        
        if aux == 1:
            sol_vect.append(0)
        
        elif aux == -1:
            sol_vect.append(1)
            
    return sol_vect


def sol_vect_to_sol(sol_vect):    
    '''transforms a vector to a readable linear combination'''
    
    basis = get_tensor_gr_gen(dim)[dim]

    aux = ''
    for i in range(len(sol_vect)):
        if sol_vect[i] == 1:
            aux += ' + '+str(basis[i])
                
    aux = aux.replace('], [',']x[')
    aux = aux.replace('] + [',' + ')
    aux = aux.replace(', ',',')
    aux = aux.replace('+','',1)
    aux = aux.replace(']]',']')
    aux = aux.replace('[[','[')
    
    return(aux)


def get_free_vars_set(exp): #to big of a method
    '''return a list with the 2^exp posible values for the free variables'''
    
    if exp >= 21:
        print('to many for an exponential method')
        return
    
    free_vars_set = []
    for i in range(2**exp):
        aux = []
        for j in reversed(bin(i)):
            if j == 'b':
                for k in range(exp-len(aux)):
                    aux.append(1)
                break
            elif j == '0':
                aux.append(1)
            elif j == '1':
                aux.append(-1)
        aux.append(-1)
        free_vars_set.append(aux)
    
    return free_vars_set

#print('set of all posible values for the free variables\n',get_free_vars_set(21))


def get_all_solutions(dim):
    '''returns a list with all vector solutions'''
      
    exp = get_num_of_free_vars_and_relations(dim)[0]
    relations = get_num_of_free_vars_and_relations(dim)[1]
    free_vars_set = get_free_vars_set(exp)
    if free_vars_set == None:
        print('too big')
        return
    
    all_solutions = []
    for free_vars in free_vars_set:
        sol_vect = free_vars_to_sol_vect(free_vars,relations)
        all_solutions.append(sol_vect)
        
    return all_solutions

#print('all solutions\n',get_all_solutions(dim))


def get_gr_transp_pairs(dim):
    '''returns the set containing the pairs of generators that are equal up to 
    transposition'''
    
    gens = get_tensor_gr_gen(dim)[dim]
    transp_pairs = []
    for gen in gens:
        neg=[gen[1],gen[0]]
        for gen2 in gens:
            if gen2 == neg and gens.index(gen) <= gens.index(gen2):
                transp_pairs.append((gens.index(gen),gens.index(gen2)))    

    
    return transp_pairs

#print('collection of basis elements related by transposition\n',get_transp_pairs(dim))


#print('collection of basis elements related by transposition\n',get_transp_pairs(dim))
#print(get_num_of_free_vars_and_relations(dim)[1])

transp_pairs = get_transp_pairs(dim)
relations = get_num_of_free_vars_and_relations(dim)[1]

for pair in transp_pairs:
    print(relations[pair[0]],relations[pair[1]])

print(relations[pair[0]].remove(2))    

