import numpy as np
import random

#inital p and q
p = 397
q = 307

#checks if prime
def isPrime(num):

    if num == 2 or num % 2 == 0:
        return False
    elif num < 2:
        return False
    for n in range(3, num):
        if num % n == 0:
            return False
    return True

#gets greatest common denominator, which would be a prime number using the numbers we have
def gcd(x,y):

    while y != 0:
        x,y = y, x%y
    return x
#function to calculate the modular inverse of e and phi to generate d
def mod_inv(e, phi_n):
    def egcd(e, phi_n):
        if e == 0:
            return (phi_n, 0, 1)
        else:
            d, x, y = egcd(phi_n % e, e)
            return (d, y - (phi_n // e) * x, x)
    d, x, _ = egcd(e, phi_n)
    if d == 1:
        return x % e
#generates e,n,d
def get_keys(p,q):

    if not isPrime(p) and not isPrime(q):
        raise ValueError('P and Q need to be prime. Both are not.')
    elif p == q:
        raise ValueError('P and Q must be different numbers. They are currently equal.')

    n = p*q
    phi_n = (p - 1) * (q-1)

    e = random.randrange(1,phi_n)
    check = gcd(e, phi_n)

    while check != 1:
        e = random.randrange(1,phi_n)
        check = gcd(e,phi_n)

    d = mod_inv(e, phi_n)

    return e, n, d
#returns text
def get_text(filename):

    with open(filename, 'r+') as f:
        message = f.read()
        print message
    f.close()
    return message
#encrypts a message
def encrypt_message(e,n):

    text = get_text('message.txt')
    text = text.split()
    cipher = []

    for x in range(len(text)):
        word = 0
        if text[x][-1] != '\n':
            text[x] += " "
        for y in range(len(text[x])):
            word += ord(text[x][y])
        cipher.append((word ** e) % n)

    with open('ciphertext.txt', 'w+') as f:
        for x in cipher:
            f.write(str(x) + ' ')
    f.close()

    print "Done."
    return cipher



def decrypt(d, n, cipherfile):

    pass


if __name__ == '__main__':

    e, n, d = get_keys(p,q)
    encrypt_message(e,n)
    print e, n, d
    #d = 37711 upon submission
