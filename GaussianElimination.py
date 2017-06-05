#Tom Kueny
#Professors Papadakis and Culver
#GE LAB with File I/O

#This lab will take a modified matrix object, perform GE on it,
#solve it as a system, determine whether the system is consistent,
#and write to a .txt file the operations it took to get the object from
#original -> echelon form ->  solved. The original matrix object will be saved
#in a .pkl file for the next execution according to whether the object is
#consistent, inconsistent, or contains only zeros.

#Program was able to solve a consistent 100x100 matrix in 8 minutes, but the .txt file
#recording the operations grew to 2.25 GB, probably because I have the program write
#every row pivot, multiplication, and addition, as well as it performing both GE
#and solving from scratch.


from numpy import *
from itertools import product
import copy
import time
import datetime
import random
import cPickle as pickle
import os.path

class OperationError(Exception):
#Exception class that will be raised if addition or multiplication cannot be
#performed or if a matrix entered that is used for arithmetic is invalid

    #Constructor will receive reason as to why Exception is raised
    def __init__(self, reason):
        self.reason = reason

    #Produces string for reason passed to constructor so that it can be output
    #to the screen
    def __str__(self):
        return repr(self.reason)

def isMatrix(matrix):
    #iterates through rows and columns, will return False if any element is
    #not of type <int>, else True
    if matrix == []:
        return True
    else:
        rows = len(matrix)
        cols = len(matrix[0])
        for row in range(rows):
            if len(matrix[row]) != cols:
                return False
            for col in range(cols):
                if type(matrix[row][col]) is not type(5.0) and matrix[row][col] != None:
                    return False
        return True

class Matrix(object):
    #constructor first tests if matrix is a matrix, then assigns appropriate
    #values to constructed Matrix object
    def __init__(self, matrix, nrows, ncols):
        validMat = isMatrix(matrix)
        try:
            if validMat:
                self.matrix = matrix
                self.nrows = nrows
                self.ncols = ncols
            else:
                raise ValueError
        except ValueError:
            print str(matrix), " is not a matrix."
    #will print a string representation since it is of type <list> of self.matrix
    def __str__(self):
        return repr(self.matrix)

    def canAdd(self, matB):
    #will test if all elements of a matrix are of the same size so that
    #addition can occur
        if self.nrows == matB.nrows and self.ncols == matB.ncols:

            for row in range(len(self.matrix)):
                for row2 in range(len(matB.matrix[row])):
                    if len(self.matrix[row]) != len(matB.matrix[row2]):
                        return False
            return True
        else:
            return False

    def canMult(self, matB):
        #tests if two matrices can be multiplied
        mat = matB.transpose()

        if len(self.matrix) != len(mat.matrix[0]):
            return False
        else:
            return True
    #Addition function to add two matrices together
    def __add__(self, matB):

        matC = Matrix([], 0, 0)
        try:
            if self.canAdd(matB):
                #iterates through rows and columns and adds respective indices together
                for row in range(len(self.matrix)):
                    #temporary list that is the sum of to matrix indices to be appended
                    #to matC.matrix
                    temp = []
                    for col in range(len(self.matrix[0])):
                        temp.append(self.matrix[row][col] + matB.matrix[row][col])
                    matC.matrix.append(temp)

                return matC
            else:
                raise OperationError("Cannot add these matrices together!")
        except OperationError as error:
            print error

    #Multiplication function for [Amatrix] * [Bmatrix]
    def __mul__(self, matB):
        if type(matB) is type(self):
            try:
                if self.canMult(matB):
                    #the product module from itertools returns a cartesian
                    #product in the form of tuples
                    from itertools import product

                    cols, rows = len(matB.matrix[0]), len(matB.matrix)
                    selfRows = range(len(self.matrix))
                    #sets a matrix that has self.nrows and matB.ncols with all
                    #values = 0
                    matC = Matrix([[0] * cols for i in selfRows], self.nrows, self.ncols)

                    for i in selfRows:
                                    #this iteration returns a cartesian product which will
                                    #give appropriate indices for the matrix,
                                    #which is in list form
                        for j, k in product(range(cols), range(rows)):
                            matC.matrix[i][j] += self.matrix[i][k] * matB.matrix[k][j]

                    return matC
                else:
                    raise OperationError("Cannot multiply these matrices together!")
            except OperationError as error:
                print error

        #this conditional will execute if self is multiplied by an integer rather
        #than a matrix object
        elif type(matB) is type(5.0):
            #empty matrix object for newly scaled matrix
            newMat = Matrix([], 0, 0)

            for row in range(len(self.matrix)):
                newMat.matrix.append([])
                for col in range(len(self.matrix[0])):
                    newMat.matrix[row].append(self.matrix[row][col]*matB)

            return newMat

    #function to transpose a matrix
    def transpose(self):

        newMat = Matrix([], 0, 0)

        #iterates through rows
        for row in range(len(self.matrix[0])):
            temp = []
            #iterates through columns
            for col in range(len(self.matrix)):
                #appends new list where rows and columns are inverted
                temp.append(self.matrix[col][row])
            newMat.matrix.append(temp)
        return newMat

    def identity(self):
        #returns the identity of a matrix (to be used if matrix**0)
        try:
            if self.canMult(self):
                matrix = [[0]*self.ncols for x in range(len(self.matrix))]
                for x in range(len(self.matrix)):
                    matrix[x][x] = 1.0
                return matrix
            else:
                raise OperationError("Matrix is not square. Cannot perform identity (matrix^0) operation.")
        except OperationError as error:
            print error
    #function to raise a matrix to desired power
    def __pow__(self, exp):
        #recursively calls method to multiply matrix by itself for exp number of
        #instances
        newMat = Matrix(self.matrix, self.nrows, self.ncols)
        newMat = self.pow_mult(newMat, exp)

        return newMat

    def pow_mult(self, mat, exp):
        #helper method for recursive call in overloading __pow__ method
        if exp == 0:
            #if exponent is 0, will return the identity of the matrix
            newMat = self.identity()
            return newMat

        elif exp == 1:
            #if matrix is raised to 1st power OR the recursive calls in the else
            #block below is decrimented to 1, this method will return the
            #original matrix object passed
            return mat

        else:
            newMat = Matrix(mat.matrix, mat.nrows, mat.ncols)
            newMat = self*mat
            return self.pow_mult(newMat,exp-1)


#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------#



class System_Equations(Matrix):

    #calls matrix class's constructor
    def __init__(self, matrix, nrows, ncols):
        Matrix.__init__(self, matrix, nrows, ncols)

    #outputs a string representation of matrix as:
    #[x,y,z] | ans
    #[x,y,z] | ans
    #...
    def __str__(self):
        matrix_table = {}
        solution_list = []
        output = ""
        for n in range(len(self.matrix)):
            solution_list.append(self.matrix[n][len(self.matrix[n])-1])
        for x in range(len(self.matrix)):
            eq_list = self.matrix[x][:len(self.matrix[x])-1]
            matrix_table[solution_list[x]] = eq_list
        for n in solution_list:
            output += str(matrix_table[n]) + " | " + str(n) + "\n"
        return output

    #retrieves values at the end of the matrix and puts them into a list
     #puts a matrix into echelon form by using the three helper functions below
    def GE(self, row, column, Gaussian = False, Solve = False):
        #sets row,column indices to zero at function call
        #loops until row/column indices reach the end of range of matrix provided

        if row < self.nrows and column < self.ncols:
            #find pivot returns a pivot value and new self object
            try:
                if self.isZeroMatrix():
                    raise ValueError
            except ValueError:
                print "Matrix provided contains all zeros! Exiting now..."
                return
            else:
                try:
                    pivot, self = self.findPivot(row, column)
                    #if operation is complete, the returning pivot will throw a TypeError
                except TypeError:
                    return self
                else:
                    pivot = 1/pivot #pivot is now inverted to be used to set this particular
                                    #row's pivot position to 1
                    self.write_operation('multiplying', pivot, row)
                    self.matrix[row] = self.rowMult(row, pivot) #row is multiplied by 1/pivot
                    self.write_matrix()
                    #once pivoted row has its target (matrix[target][target]) set to 1
                    #following loop will multiply this row by a needed factor specified
                    #by the indices that reveal a non-zero value in a lower row, and this
                    #temporary row will be added to the lower row so that in the target column
                    #it will have the blow values set to zero
                    if Gaussian:
                        for ROW in range(row, self.nrows):
                            if ROW == row:  #iteration will skip the current row GE is called on
                                continue
                            elif self.matrix[ROW][column] != 0:     #if target column in other rows is not 0
                                factor = self.matrix[ROW][column]   #this iteration will multiply the row by
                                factor = -(factor)
                                self.write_operation('adding', factor, row, ROW)  #the negative value of the iterated row and target
                                tempRow = self.rowMult(row, factor) #column, set that to a tempRow, and then add that to the current row
                                self.matrix[ROW] = self.rowAdd(ROW, tempRow) #to get zero
                                self.write_matrix()
                    #once this is done successfully, row/column indices are incremented and loop executes again
                    elif Solve:
                        for ROW in range(0, self.nrows):
                            if ROW == row:  #iteration will skip the current row GE is called on
                                continue
                            elif self.matrix[ROW][column] != 0:     #if target column in other rows is not 0
                                factor = self.matrix[ROW][column]   #this iteration will multiply the row by
                                factor = -(factor)
                                self.write_operation('adding', factor, row, ROW)  #the negative value of the iterated row and target
                                tempRow = self.rowMult(row, factor) #column, set that to a tempRow, and then add that to the current row
                                self.matrix[ROW] = self.rowAdd(ROW, tempRow) #to get zero
                                self.write_matrix()
                    row += 1
                    column += 1
                #returns incremented row and column and will call the GE method recursively
                return self.GE(row, column, Gaussian, Solve)
        else:
            return self

    #helper function uses numpy.multiply and numpy.array.tolist() to yield a
    #list multiplied by a scalar factor
    def rowMult(self, row, factor):
        return multiply(self.matrix[row], factor).tolist()

    #uses numpy.add and numpy.array.tolist() to add two rows together to make a new rowAdd
    def rowAdd(self, row, otherRow):
         return add(self.matrix[row],otherRow).tolist()

    #finds next non-zero value in specific column and brings it to the highest needed postion in
    #the matrix
    def findPivot(self, row, col):
        #if matrix[row][col] contains 0 value, will iterate through rows at the
        #specified column until a non-zero value is found in that column
        NON_ZERO_FOUND = False
        if self.matrix[row][col] == 0:
            ZEROHERE = row #ZEROHERE is the base value for row indices
            while ZEROHERE < self.nrows - 1 and self.matrix[ZEROHERE][col] == 0:
                ZEROHERE += 1 #searches through next row, breaks if non-zero is found

            #BLOCK TO CHECK IF THERE IS AN ALL ZERO COLUMN || MATRIX
            if self.matrix[ZEROHERE][col] != 0:
                NON_ZERO_FOUND = True
            if not(NON_ZERO_FOUND):
                if self.matrix[ZEROHERE][col] == 0:
                    for c in range(len(self.matrix[ZEROHERE])):
                        if self.matrix[ZEROHERE][c] != 0:
                            pivot = None
                            return pivot, self
                if row + 1 == self.nrows and not(self.isZeroMatrix):
                    pivot = None
                    return pivot, self
                elif self.isZeroMatrix():
                    print "We should have covered this..."
                elif col+1 == self.ncols and row+1 != self.nrows:
                    return self.findPivot(row+1, col) #recursively calls this method to find a non-zero value in next column
            #block that handles a non-zero value in target column
            else:
                factor = None
                self.write_operation('pivoting', factor, row, ZEROHERE)
                tempList = self.matrix[row] #creates temporary list for the row needing a pivot
                self.matrix[row] = self.matrix[ZEROHERE] #sets the highest row that does not yet have a pivoted value to the next non zero row in the specified column
                self.matrix[ZEROHERE] = tempList #moves original row to new, lower position

                pivot = float(self.matrix[row][col]) #pivot value is set to be used as a factor/scalar
                #ensures pivot value is a float
                #double checks that if pivot value is zero, no zero-division will occur

                return pivot, self #tuple of pivot and matrix object is returned
        elif self.matrix[row][col] != 0:
            pivot = float(self.matrix[row][col])
            return pivot, self

    def isZeroMatrix(self):
        ZERO_MATRIX = True
        for a in range(0,self.nrows):
            for b in range(0, self.ncols):
                if self.matrix[a][b] != 0:
                    ZERO_MATRIX = False
                    break
                else:
                    continue
        return ZERO_MATRIX

    def consistent(self):
        unknowns = self.ncols - 1
        consistent = True
        solutions = []
        for x in range(0,unknowns):
            if self.matrix[x][x] != 0:
                solutions.append("x%s = %s" % (str(x), str(self.matrix[x][-1])))
                continue
            elif self.matrix[x][x] == 0 and self.matrix[x][-1] != 0:
                consistent = False
                solutions = "The matrix is inconsistent."
                break
            elif self.matrix[x][x] == 0 and self.matrix[x][-1] == 0:
                print "Found a non-zero row."
                continue
        if self.nrows > unknowns:
            for x in range(unknowns, self.nrows):
                if self.matrix[x][-1] != 0:
                    consistent = False
                    break
        return consistent, solutions

    def write_operation(self, operation, factor = None, x = None,  y = None):

        op_no_ing = operation[:-3]
        with open('matrix_ops.txt', 'a') as f:
            print "Performing row %s operation..." % op_no_ing
            if operation == 'multiplying' and factor != None:
                data = "%s %s by %s...\n" % (operation, str(factor), str(self.matrix[x]))
                f.write(data)
            elif operation == 'adding' and (x and y and factor) != None:
                data = "%s %s times %s to %s...\n" % (operation, str(factor), str(self.matrix[x]), str(self.matrix[y]))
                f.write(data)
            elif operation == 'pivoting':
                data = "%s %s and %s\n" % (operation, str(self.matrix[x]), str(self.matrix[y]))
            f.close()
        return

    def write_matrix(self, original = False, final = False, GE = False):

        with open('matrix_ops.txt', 'a') as f:
            if original:
                print "The original matrix is being written in 'matrix_file.txt':\n\n", self
                try:
                    f.write("The original matrix is...\n%s\n" % self)
                except:
                    print "Error in writing matrix. Please restart program."
            elif final:
                print "The final matrix is being written in 'matrix_ops.txt':\n\n", self
                try:
                    f.write("The solved matrix is...\n%s\n" % self)
                except:
                    print "Error in writing matrix. Please restart program."
            elif GE:
                print "The matrix after Gaussian Elimination is being written in 'matrix_file.txt':\n\n", self
                try:
                    f.write("The GE matrix is...\n%s\n" % self)
                except:
                    print "Error in writing matrix. Please restart program."
            else:
                print "The resulting matrix is being written in 'matrix_ops.txt':\n\n", self
                try:
                    f.write("The resulting matrix is...\n%s\n" % self)
                except:
                    print "Error in writing matrix. Please restart program."
            f.close()
        return

def input_matrix(rows, cols):

    matrix = []
    for x in range(0, rows):
        print "This is for equation %s...\n" % str(x)
        temp = []
        for y in range(0, cols):
            num = float(raw_input("Enter a value for %s,%s: " % (str(x),str(y))))
            try:
                if type(num) != type(5.0):
                    raise IOError
            except:
                print "Invalid input. Let's try again, shall we?\n"
                return input_matrix(rows, cols)
            else:
                temp.append(num)
        matrix.append(temp)
    return matrix

def generate_matrix(rows, cols, max, min):
    matrix = []
    for x in range(0, rows):
        temp = []
        for y in range(0, cols):
            num = float(random.randint(min,max))
            temp.append(num)
        matrix.append(temp)
    return matrix

def welcome_past():
    name = raw_input("What is your name?  ")
    name = name[0].upper() + name[1:]

    choice = "generate a past matrix "

    past_matrix = generate_past_matrix(name)

    return name, choice, past_matrix.nrows, past_matrix.ncols, past_matrix

def welcome_new():
    name = raw_input("What is your name?  ")
    name = name[0].upper() + name[1:]
    print "Hello, " + name + "!\nWould you like to input a matrix to perform GE and then solve it?"
    print "\n\nOr would you like for one to be randomly generated for you?\n\n"
    input_or_rand = raw_input("Write 'input' or 'random' on the line below:\n")

    try:
        if input_or_rand != 'input' and input_or_rand != 'random':
            raise IOError
    except IOError:
        #recursively call main again?
        print "You must have mistyped, let's try again.\n"
        return welcome()
    else:
        variables = raw_input("How many unknowns are there?")
        equations = raw_input("How many equations?")
        rows = int(equations)
        cols = int(variables) + 1


        if input_or_rand == 'input':
            matrix = input_matrix(rows, cols)

        elif input_or_rand == 'random':
            matrix = generate_matrix(rows, cols)
            input_or_rand = 'generate a ' + input_or_rand


    return name, input_or_rand, rows, cols, matrix

def generate_past_matrix(name):
    choice = raw_input("Hello, " + name + "!\nWould you like to generate a consisent, inconsistent, or zero matrix?\n(C, I, or Z): ")
    choice = choice.upper()
    try:
        if choice != 'C' and choice != 'I' and choice != 'Z':
            raise IOError
    except IOError:
        print "That is not a choice!"
        generate_past_matrix(name)
    else:
        if choice == 'C' and os.path.isfile('consistent_matrices.pkl'):
            with open('consistent_matrices.pkl', 'rb') as f:
                past_matrix = pickle.load(f)
                f.close()
        elif choice == 'I' and os.path.isfile('inconsistent_matrices.pkl'):
            with open('inconsistent_matrices.pkl', 'rb') as f:
                past_matrix = pickle.load(f)
                f.close()
        elif choice == 'Z' and os.path.isfile('zero_matrices.pkl'):
            with open('zero_matrices.pkl', 'rb') as f:
                past_matrix = pickle.load(f)
                f.close()
        else:
            print "Oops, looks like the file containing the matrix you want doesn't exist yet...\n"
            print "Create your choice of matrix so we can prevent this in the future!\n"
            sys.exit(0)
        return past_matrix

def save_consistent_matrix(UNIQUE_MATRIX_NAME):
    written = False
    if not(os.path.isfile('consistent_matrices.pkl')):
        with open('consistent_matrices.pkl', 'wb') as f:
            pickle.dump(UNIQUE_MATRIX_NAME, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        written = True
        del UNIQUE_MATRIX_NAME
    else:
        with open('consistent_matrices.pkl', 'wb') as f:
            pickle.dump(UNIQUE_MATRIX_NAME, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        written = True
        del UNIQUE_MATRIX_NAME
    try:
        if not(written):
            raise IOError
    except IOError:
        print "Error in attempting to save matrix object..."
    else:
        print "successfully saved matrix object..."

def save_zero_matrix(UNIQUE_MATRIX_NAME):
    written = False
    if not(os.path.isfile('zero_matrices.pkl')):
        with open('zero_matrices.pkl', 'wb') as f:
            pickle.dump(UNIQUE_MATRIX_NAME, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        written = True
        del UNIQUE_MATRIX_NAME
    else:
        with open('zero_matrices.pkl', 'wb') as f:
            pickle.dump(UNIQUE_MATRIX_NAME, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        written = True
        del UNIQUE_MATRIX_NAME
    try:
        if not(written):
            raise IOError
    except IOError:
        print "Error in attempting to save matrix object..."
    else:
        print "successfully saved matrix object..."


def save_inconsistent_matrix(UNIQUE_MATRIX_NAME):
    written = False
    if not(os.path.isfile('inconsistent_matrices.pkl')):
        with open('inconsistent_matrices.pkl', 'wb') as f:
            pickle.dump(UNIQUE_MATRIX_NAME, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        written = True
        del UNIQUE_MATRIX_NAME
    else:
        with open('inconsistent_matrices.pkl', 'wb') as f:
            pickle.dump(UNIQUE_MATRIX_NAME, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        written = True
        del UNIQUE_MATRIX_NAME
    try:
        if not(written):
            raise IOError
    except IOError:
        print "Error in attempting to save matrix object..."
    else:
        print "successfully saved matrix object..."


if __name__ == '__main__':
    import sys
    import timeit

    with open('matrix_ops.txt', 'w+') as f:
        f.write("Program started at " + datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S\n"))
        f.close()
    prompt = raw_input("Generate new or past matrix? (N/P): ")
    prompt = prompt.upper()
    if prompt == 'N':
        name, input_or_rand, rows, cols, lst = welcome_new()
        mat = System_Equations(lst, rows, cols)
        with open('matrix_ops.txt', 'a+') as f:
            f.write("%s has chosen to %s a matrix to perform GE and solve.\n" % (name, input_or_rand))
            f.write("The matrix is a %sx%s matrix.\n" % (str(rows),str(cols)))
            f.close()


    elif prompt == 'P':
        name, input_or_rand, rows, cols, mat = welcome_past()
        with open('matrix_ops.txt', 'a+') as f:
            f.write("%s has chosen to %s a matrix to perform GE and solve.\n" % (name, input_or_rand))
            f.write("The matrix is a %sx%s matrix.\n" % (str(rows),str(cols)))
            f.close()


    mat.write_matrix(True, False)
    if mat.isZeroMatrix():
        with open('matrix_ops.txt', 'a+') as f:
            f.write("The given matrix contains all zeros. We're done here.\n")
            f.write("Program finished at " + datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S\n"))
            f.close()
        cpy3 = copy.copy(mat)
        save_zero_matrix(cpy3)
        sys.exit(0)

    solvedMat = copy.deepcopy(mat)
    GE_Mat = copy.deepcopy(mat)

    row, col = 0,0

    with open('matrix_ops.txt', 'a+') as f:
        f.write("Performing Gaussian Elimination...\n")
        f.close()
    time_start_GE = timeit.timeit()
    GE_Mat = GE_Mat.GE(row, col, True, False)
    time_stop_GE = timeit.timeit()

    GE_time = time_stop_GE - time_start_GE

    row, col = 0,0

    with open('matrix_ops.txt', 'a+') as f:
        f.write("Solving the matrix...\n")
        f.close()

    time_start_solve = timeit.timeit()
    solvedMat = solvedMat.GE(row, col, False, True)
    time_stop_solve  = timeit.timeit()
    solve_time = time_stop_solve - time_start_solve

    mat.write_matrix(True)
    GE_Mat.write_matrix(False, False, True)
    solvedMat.write_matrix(False, True, False)

    isConsistent, solutions = solvedMat.consistent()

    if isConsistent:
        print "The matrix is consistent."
        with open('matrix_ops.txt', 'a+') as f:
            f.write("The following matrix is consistent:\n\n%s" % mat)
            f.write("Solutions are:\n")
            for s in solutions:
                f.write("For %s\n" % s)
            f.close()
        cpy = copy.copy(mat)
        save_consistent_matrix(cpy)
    elif not(isConsistent):
        print "The matrix is inconsistent."
        with open('matrix_ops.txt', 'a+') as f:
            f.write("The following matrix is inconsistent:\n\n%s" % mat)
            f.write("Solutions are:\n")
            for s in solutions:
                f.write("%s\n" % s)
            f.close()
        cpy2 = copy.copy(mat)
        save_inconsistent_matrix(cpy2)


    with open('matrix_ops.txt', 'a+') as f:
        f.write("Time took to put into GE = %s\nTime took to solve = %s" % (str(GE_time), str(solve_time)))
        f.write("Program finished at " + datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S\n"))
        f.close()
