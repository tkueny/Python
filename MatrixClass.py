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
