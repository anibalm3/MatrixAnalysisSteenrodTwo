import numpy as np
import itertools

def get_gr_gen(dim):
    '''returns the list whose d-entry is the lexicographically ordered list of 
    d-faces of the (dim)-simplex'''  

    gr_gen = []
    for i in range(dim+1):
        els = [list(x) for x in itertools.combinations(set(range(dim+1)), i+1)]
        gr_gen.append(els)
    
    return gr_gen

def get_gr_partial(dim): 
    '''returns the list whose d-entry is the np.array (matrix) representing the boundary map from d to (d-1)-chains 
    of the standard (dim)-simplex with bases lexicographically ordered'''

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
                        partial_sub_degree[j,i]=(-1)**vertex
                        break
        
        gr_partial.append(partial_sub_degree)
    
    return(gr_partial)
    
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
    

def get_gr_dim(gr_obj):
    '''returns the lenghts of the lists in a list'''
    
    gr_dim = []    
    for l in gr_obj:
        gr_dim.append(len(l))

    return gr_dim
                

def get_tensor_partial(dim):
    '''computes the matrix representing the dim-boundary in the tensor product 
    (with respect to the canonical basis lexicografically ordered)'''

    gr_dim = get_gr_dim(get_gr_gen(dim))
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
        
        for j in range(dim):
            ver_len = gr_dim[j]*gr_dim[dim-1-j]
    
            if j+1 == i: #dx1:
                identity = np.eye(gr_dim[dim-i],dtype=int)
                block = np.kron(gr_partial[i],identity)
                
            elif dim-j == dim-i: #1xd
                identity = np.eye(gr_dim[i],dtype=int)
                block = np.kron(identity,gr_partial[dim-i])
                
            else: 
                block = np.zeros((ver_len,hor_len),dtype=int)
            ver_stack = np.vstack((ver_stack,block))
            
        hor_stack = np.hstack((hor_stack,ver_stack))
    
    #deleting first row and column
    hor_stack = np.delete(hor_stack, (0), axis=0)    
    out_matrix = np.delete(hor_stack, (0), axis=1)    

    return out_matrix
    

def get_partial(face):
    '''given a face it returns the lexicographically 
    order list of generators appearing in its boundary'''

    partial = []
    for i in face:
        di_face = list(face)
        di_face.pop(i)
        partial.append([(-1)**i,di_face])

    return(partial)

def get_diag(sign,face):
    '''given a signed face, [plus/minus 1, [u,v,...,w]], it returns the 
    list of signed generators, [plus/minus 1, [[u,v],[v,...,w]]], appearing 
    in its standard diagonal'''
    
    diag = []
    for i in face:
        diag.append([sign,face[:face.index(i)+1],face[face.index(i):]])

    return(diag)


def get_diag_partial(dim):
    '''returns the vector, with entries: -1, 0 or 1, representing the linear combination 
    obtained by taking boundary and then the standard diagonal'''
    
    simplex = list(range(dim+1))    
    partial = get_partial(simplex)
    
    diag_partial = []
    for face in partial: #computes all signed terms in the diagonal of the boundary
        aux = get_diag(face[0],face[1])
        diag_partial.extend(aux)
        
    tensor_gen = get_tensor_gr_gen(dim)[dim-1]    
    out_vector = np.zeros((len(tensor_gen),1),dtype=int)    
    
    for gen in tensor_gen: #constructs the vector representing the linear combination
        for term in diag_partial:
            if gen == term[1:]:
                out_vector[tensor_gen.index(gen)] = term[0]
    
    return out_vector


dim = 2
#print(get_tensor_partial(dim))
print(get_diag_partial(dim))

from sympy import Matrix

P = Matrix(get_tensor_partial(dim))
print(P)
P = P.col_insert(4, Matrix(get_diag_partial(dim)))
print(P)

print(len(P.columnspace()))
print(P.rref())        
